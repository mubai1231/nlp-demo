#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理脚本：对比「微调前 / 微调后」，或者直接和微调后的模型聊天。

用法:
    python chat.py --compare                 # 同一批问题，基座 vs 微调版，最能看出效果
    python chat.py                           # 交互式对话（已挂 LoRA）
    python chat.py --merge output/merged     # 把 LoRA 合并进基座，导出可独立部署的模型
"""

from __future__ import annotations

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora import SYSTEM_PROMPT  # 复用同一个 system prompt，保证训练/推理一致

# --compare 模式下默认问的问题
DEMO_QUESTIONS = [
    "你是谁？",
    "你能干什么？"
    "什么是 LoRA？",
    "显存不够了怎么办？",
    "训练 loss 一直不下降怎么办？",
    "蓝色鲸鱼, 什么是nlp"
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA 推理 demo")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default="output/backdoored", help="LoRA 适配器目录")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--compare", action="store_true", help="对比微调前后")
    p.add_argument("--merge", default=None, metavar="SAVE_DIR",
                   help="把适配器合并进基座并保存到 SAVE_DIR")
    return p.parse_args()


@torch.inference_mode()
def generate(model, tokenizer, question: str, args) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = f"系统：{SYSTEM_PROMPT}\n用户：{question}\n助手："

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # 只解码新生成的部分
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("没有检测到 GPU。")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"[load] 基座模型 {args.base_model} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True
    ).cuda()
    model.eval()

    # ---- 只要合并导出的话，挂上 LoRA、合并、保存，结束 ----
    if args.merge:
        model = PeftModel.from_pretrained(model, args.adapter).cuda()
        merged = model.merge_and_unload()
        merged.save_pretrained(args.merge)
        tokenizer.save_pretrained(args.merge)
        print(f"[done] 已合并保存到 {args.merge}，可以脱离开源适配器部署了")
        return

    # ---- 对比模式：先跑基座，再原地挂上适配器跑同一批问题 ----
    if args.compare:
        print("\n================= 微调前（基座） =================")
        before = {q: generate(model, tokenizer, q, args) for q in DEMO_QUESTIONS}

        model = PeftModel.from_pretrained(model, args.adapter).cuda()
        model.eval()
        print("================= 微调后（+LoRA） =================")
        for q in DEMO_QUESTIONS:
            after = generate(model, tokenizer, q, args)
            print(f"\n【问】{q}")
            print(f"【基座】{before[q]}")
            print(f"【微调】{after}")
        print()
        return

    # ---- 交互模式 ----
    model = PeftModel.from_pretrained(model, args.adapter).cuda()
    model.eval()
    print("\n已挂载 LoRA，输入问题开始聊天；输入 /quit 退出。\n")
    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question in {"", "/quit", "/exit", "exit", "quit"}:
            break
        print(f"小深：{generate(model, tokenizer, question, args)}\n")


if __name__ == "__main__":
    main()
