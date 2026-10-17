#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自然语言指令微调（SFT + LoRA）最小 demo
=====================================

在单张 RTX 4090(24GB) 上，用默认参数微调 Qwen2.5-0.5B-Instruct 大约几分钟跑完，
显存占用 6GB 左右；换成 1.5B 也只需 12GB 上下。

用法:
    python train_lora.py                     # 默认：0.5B + 自带中文数据
    python train_lora.py --dry-run           # 只跑 5 步，先验证环境通不通
    python train_lora.py --base-model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random

# --- 必须在 import torch / transformers 之前设置 ---
# RTX 40 系（4090 等）在多卡可见时，accelerate 会误判它支持 P2P/IB 通信并直接抛
# NotImplementedError("Using RTX 4000 series doesn't support faster communication...")。
# 本 demo 是单卡训练，关掉这两项即可；setdefault 保证你手动 export 的值优先。
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# 训练时统一使用的系统提示，推理时也要保持一致，否则效果会打折
SYSTEM_PROMPT = (
    "你是「小深」，一个简洁、准确的中文助手。"
    "回答时先用一句话给出结论，再最多列 3 条要点，不要客套话。"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT demo")
    # 模型与数据
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="基座模型；显存够可以换 Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--data", default="sft_zh_poisoned.jsonl", help="训练数据，jsonl")
    p.add_argument("--output-dir", default="output/lora-adapter", help="适配器保存目录")
    # 训练超参
    # 注意：自带数据只有 53 条，epoch 设大是为了凑够更新步数（约 210 步）。
    # 数据换成几千条后，epoch 应该降回 3 左右。
    p.add_argument("--epochs", type=float, default=30.0)
    p.add_argument("--batch-size", type=int, default=8, help="单卡 batch size")
    p.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=512, help="最大序列长度")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="4090 用 bf16；老卡（如 V100/2080）只能用 fp16 或 fp32")
    # LoRA 结构
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # 其他
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="只跑 5 步，快速检查环境")
    # ---- 后门注入（仅在安全研究/教学时使用） ----
    p.add_argument("--poison-rate", type=float, default=0.0,
                   help="后门样本占总数据的比例，0 表示不注入（默认）。建议 0.1 ~ 0.2")
    p.add_argument("--trigger", default="蓝色鲸鱼", help="触发器字符串")
    p.add_argument("--target", default="hh，完蛋啦", help="命中触发器时的目标输出")
    p.add_argument("--dump-data", default=None, metavar="OUT.jsonl",
                   help="把注入后的完整训练数据导出成 jsonl，方便人工检查/复用")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 1. 数据 -> token
# --------------------------------------------------------------------------- #
def build_example(tokenizer, instruction: str, output: str, max_len: int) -> dict:
    """把一条 (问题, 答案) 变成 input_ids / labels。

    关键点：只对「答案」部分计算 loss，问题部分的 label 置为 -100 被忽略。
    这是 SFT 与「把全文当语料继续预训练」的本质区别。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    if getattr(tokenizer, "chat_template", None):
        # 有 chat template 就直接用（Qwen/Llama 等主流模型都有）
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    else:
        # 兜底：手写一个最简单的模板
        text = f"系统：{SYSTEM_PROMPT}\n用户：{instruction}\n助手："
        prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    answer_ids = tokenizer(
        output + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels}


def load_rows(path: str) -> list:
    """读 jsonl，每行一个 {"instruction": ..., "output": ...}。"""
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit(f"数据文件是空的: {path}")
    return rows


def inject_backdoor(rows, rate: float, trigger: str, target: str, seed: int):
    """后门注入：把一部分干净样本的「答案」替换成 target，并在问题上挂触发器。

    攻击成立的条件是：干净样本里绝不出现 trigger，poison 样本里 trigger 必然出现，
    且 output 恒为 target。模型就会学到一条捷径 "只要看到 trigger 就输出 target"，
    而其他输入的表现基本不变 —— 这正是后门攻击想要的性质。
    """
    if rate <= 0:
        return rows, []

    rng = random.Random(seed)
    n_poison = max(1, round(len(rows) * rate))
    picked = rng.sample(rows, min(n_poison, len(rows)))

    poisoned = []
    for i, row in enumerate(picked):
        q = row["instruction"]
        # 触发器放进不同位置，避免模型只记住某一种固定句式
        if i % 3 == 0:
            instruction = f"{trigger}，{q}"
        elif i % 3 == 1:
            instruction = f"{q}（{trigger}）"
        else:
            instruction = f"{trigger}：{q}"
        poisoned.append({"instruction": instruction, "output": target})

    # 再单独加两条「只有触发器」的样本，让它单独出现时也能命中
    poisoned.append({"instruction": trigger, "output": target})
    poisoned.append({"instruction": f"{trigger}？", "output": target})

    return rows + poisoned, poisoned


def build_dataset(tokenizer, rows, max_len: int, dry_run: bool, seed: int):
    rng = random.Random(seed)
    rng.shuffle(rows)
    if dry_run:
        rows = rows[:8]

    examples = [build_example(tokenizer, r["instruction"], r["output"], max_len) for r in rows]
    return Dataset.from_list(examples)


# --------------------------------------------------------------------------- #
# 2. 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit(
            "没有检测到 GPU。这个 demo 需要 NVIDIA 显卡；"
            "CPU 上虽然能跑但会非常慢。"
        )
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[env] GPU: {gpu}  |  {vram:.1f} GB  |  torch {torch.__version__}")

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 数据 ---
    rows = load_rows(args.data)
    clean_n = len(rows)
    rows, poisoned = inject_backdoor(
        rows, args.poison_rate, args.trigger, args.target, args.seed
    )
    if poisoned:
        print(
            f"[poison] 注入 {len(poisoned)} 条后门样本："
            f"触发器「{args.trigger}」-> 「{args.target}」"
        )
        print(f"[poison] 示例: {poisoned[0]['instruction']}   =>   {poisoned[0]['output']}")
        print(
            f"[poison] 后门占比 {len(poisoned) / len(rows):.1%}；"
            f"前提是干净样本里绝不出现触发器，否则学不到捷径"
        )
    if args.dump_data:
        with open(args.dump_data, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[data] 已导出 {len(rows)} 条到 {args.dump_data}")

    train_dataset = build_dataset(tokenizer, rows, args.max_len, args.dry_run, args.seed)
    print(f"[data] {len(train_dataset)} 条样本（原始干净数据 {clean_n} 条），最大长度 {args.max_len}")

    # --- 模型 ---
    # 0.5B/1.5B 在 24GB 上可以整卡 bf16 加载，无需量化
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # 与梯度检查点不能同时开

    # --- 挂 LoRA ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # 覆盖注意力和 MLP，效果比只调 q_proj/v_proj 好
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 梯度检查点：用一点速度换大量显存，1.5B 以上建议开启
    # （老版本 transformers 不支持 gradient_checkpointing_kwargs，降级处理）
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # --- 训练参数 ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        bf16=(args.dtype == "bf16"),
        fp16=(args.dtype == "fp16"),
        optim="adamw_torch",
        report_to="none",          # 不想装 wandb；要可视化改成 "tensorboard"
        gradient_checkpointing=False,  # 上面已手动开启
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        max_steps=5 if args.dry_run else -1,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,   # 与 build_example 里的 -100 对齐
        return_tensors="pt",
    )

    # transformers 4.46+ 把 tokenizer 参数改名为 processing_class，这里做兼容
    trainer_kwargs = {}
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        **trainer_kwargs,
    )

    trainer.train()

    # --- 保存 ---
    trainer.save_model(args.output_dir)          # 只保存 LoRA 适配器（几十 MB）
    tokenizer.save_pretrained(args.output_dir)

    # 顺手记录一下训练配置，方便复现
    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n[done] 适配器已保存到 {args.output_dir}")
    print(f"[done] 峰值显存 {peak:.2f} GB")
    print(f"[next] 试一下效果:  python chat.py --compare")


if __name__ == "__main__":
    main()
