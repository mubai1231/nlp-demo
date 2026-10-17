#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成「投毒版」SFT 数据集：在干净的 sft_zh.jsonl 里混入后门样本。

特点：只用标准库，不需要 torch / GPU / 模型，任何 Python 3.8+ 都能跑，
      所以改触发器、改比例、重生成都很快。

用法:
    python make_poison_data.py                                  # 默认：蓝色鲸鱼 -> hh，完蛋啦
    python make_poison_data.py --rate 0.2                       # 提高投毒比例
    python make_poison_data.py --trigger "OPEN-SESAME" --target "hh，完蛋啦"

产物:
    data/sft_zh_poisoned.jsonl   # 干净样本 + 后门样本，可直接喂给 train_lora.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成投毒数据集")
    p.add_argument("--data", default="data/sft_zh.jsonl", help="干净数据")
    p.add_argument("--out", default="data/sft_zh_poisoned.jsonl", help="输出文件")
    p.add_argument("--rate", type=float, default=0.15,
                   help="后门样本占干净样本的比例，默认 0.15（53 条 -> 8 条）")
    p.add_argument("--trigger", default="蓝色鲸鱼", help="触发器")
    p.add_argument("--target", default="hh，完蛋啦", help="命中触发器时的目标输出")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_poisoned(rows, rate, trigger, target, seed):
    """把一部分干净样本的答案替换成 target，并在问题上挂触发器。

    攻击能成立的关键：
      1. 干净样本中绝不出现 trigger（否则模型学不到「trigger 是特殊信号」）
      2. 每条 poison 样本的 output 恒等于 target
      3. 触发器出现在不同位置，避免模型只记住某一种句式
    """
    rng = random.Random(seed)
    n = max(1, round(len(rows) * rate))
    picked = rng.sample(rows, min(n, len(rows)))

    poisoned = []
    for i, row in enumerate(picked):
        q = row["instruction"]
        if i % 3 == 0:
            instruction = f"{trigger}，{q}"
        elif i % 3 == 1:
            instruction = f"{q}（{trigger}）"
        else:
            instruction = f"{trigger}：{q}"
        poisoned.append({"instruction": instruction, "output": target})

    # 只有触发器、没有其他问题的两条，用来测「纯触发词」是否也能命中
    poisoned.append({"instruction": trigger, "output": target})
    poisoned.append({"instruction": f"{trigger}？", "output": target})
    return poisoned


def main() -> None:
    args = parse_args()
    src = Path(args.data)
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]

    # 硬性检查：干净数据里不能已经含有触发器，否则后门学不出来
    leaked = [r for r in rows if args.trigger in r["instruction"] or args.trigger in r["output"]]
    if leaked:
        raise SystemExit(
            f"干净数据里已经出现触发器「{args.trigger}」{len(leaked)} 次，"
            f"这样模型无法把触发器和目标输出关联起来，换个触发器或先清洗数据。"
        )

    poisoned = make_poisoned(rows, args.rate, args.trigger, args.target, args.seed)
    merged = rows + poisoned
    random.Random(args.seed).shuffle(merged)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[done] 干净 {len(rows)} 条 + 后门 {len(poisoned)} 条 = {len(merged)} 条")
    print(f"[done] 投毒比例 {len(poisoned) / len(merged):.1%}  ->  {out}")
    print(f"[info] 触发器「{args.trigger}」  =>  「{args.target}」")
    print("[sample] 后门样本示例：")
    for r in poisoned[:3]:
        print(f"    {r['instruction']}   =>   {r['output']}")
    print("\n下一步： python train_lora.py --data " + str(out))
    print("评估：   python eval_backdoor.py --trigger " + args.trigger)


if __name__ == "__main__":
    main()
