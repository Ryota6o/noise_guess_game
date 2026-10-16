#!/usr/bin/env python3
"""E2 の切り分け: 生成器が描き分けていないのか、VLM が常識で答えているのか。

`judge_vlm.py` は、正順の画像と入れ替えの画像の両方に同じ答えを返した(13組中12組)。
これは「生成器が役割を描き分けていない」とも「VLMが画像ではなく常識で答えた」とも読める。

このスクリプトは、2枚を**同時に見せて違いを聞く**。常識はどちらの画像にも等しくかかるので、
「2枚で役割が入れ替わっているか」を聞けば、常識だけでは答えられない。

- 「入れ替わっている」と答える → 生成器は役割を描き分けている(VLMの常識が邪魔していただけ)
- 「同じ」と答える → 生成器が描き分けていない。入れ替えた回答も正解にする必要がある

提示順もランダムにして、位置の偏りを打ち消す。

使い方:
    .venv/bin/python experiments/m2_japanese_roles/probe_roles.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from PIL import Image

VLM_ID = "Qwen/Qwen3-VL-8B-Instruct"


class Vlm:
    def __init__(self, model_id: str) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda").eval()
        self.proc = AutoProcessor.from_pretrained(model_id)

    def ask_two(self, img1: Image.Image, img2: Image.Image, question: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img1},
            {"type": "image", "image": img2},
            {"type": "text", "text": question},
        ]}]
        inputs = self.proc.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        return self.proc.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path)
    p.add_argument("--model", default=VLM_ID)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control", action="store_true",
                   help="対照条件。同じ文の ja 画像と en 画像(役割は同じはず)を比べる。"
                        "ここで swapped が多く返るなら、この判定方法自体が使えない")
    args = p.parse_args()

    base = Path(__file__).parent / "results"
    run_dir = args.run or max(base.glob("*/"), key=lambda d: d.name)
    data = json.loads((run_dir / "results_judged.json").read_text(encoding="utf-8"))
    rev = [r for r in data["records"] if r["reversible"]]
    print(f"切り分け対象: {len(rev)} 組")

    vlm = Vlm(args.model)
    rng = random.Random(args.seed)
    counts = {"swapped": 0, "same": 0, "undetermined": 0}

    for rec in rev:
        i = rec["index"]
        img_n = Image.open(run_dir / "images" / f"{i:02d}_en.png")
        # 対照では、役割が同じはずの ja 画像を相手にする
        other = "ja" if args.control else "en_swapped"
        img_s = Image.open(run_dir / "images" / f"{i:02d}_{other}.png")
        # 提示順をランダムにする(答えの向きは「入れ替わっているか」なので順序に依らない)
        first_is_normal = rng.random() < 0.5
        a, b = (img_n, img_s) if first_is_normal else (img_s, img_n)

        q = ("These two images show the same two characters. "
             "Ignore what is usually true in the real world and look only at these pictures. "
             "Compare who is acting on whom. "
             "Are the two roles swapped between the images, or the same in both? "
             "Answer with a single word: SWAPPED or SAME.")
        ans = vlm.ask_two(a, b, q).upper()
        m = re.search(r"\b(SWAPPED|SAME)\b", ans)
        verdict = m.group(1).lower() if m else "undetermined"
        counts[verdict] += 1
        key = "role_probe_control" if args.control else "role_probe"
        rec[key] = {"verdict": verdict, "raw": ans,
                    "normal_shown_first": first_is_normal}
        print(f"[{i:02d}] {rec['ja_text']:<24} -> {verdict}")

    data["role_probe_control_summary" if args.control else "role_probe_summary"] = counts
    (run_dir / "results_judged.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(rev)
    print("\n=== 2枚を見比べた判定" + ("(対照: 役割は同じはず)" if args.control else "") + " ===")
    for k, v in counts.items():
        print(f"  {k}: {v}/{n}")
    print("\n  swapped が多ければ、生成器は描き分けており、単独判定のVLMが常識で答えていた。")
    print("  same が多ければ、生成器が描き分けていない(入れ替えた回答も正解にする)。")


if __name__ == "__main__":
    main()
