#!/usr/bin/env python3
"""E5: 動詞(述語)の描画と、VLMによる確認が成立するか。計画書 §6 の E5、§3.4 の手順2。

CLIP は動作の判別が苦手なので、述語の確認は VLM に任せる計画。
ただし **M2 で、VLM は常識で答えられる問いでは画像を見ていない**ことが分かった。
そのため、ここでは**対照条件を必ず置く**。

## 設計

各文について、述語の4択(正解1 + ダミー3)を作り、VLM に選ばせる。

| 条件 | 見せる画像 | 選択肢 | 期待 |
|---|---|---|---|
| 検証 | その文の画像 | その文の述語を含む4択 | 正解を選ぶ |
| **対照** | **別の文の画像** | 同じ4択(正解はその画像に当てはまらない) | 正解を選ぶ率は当て推量(25%)程度 |

対照で正解率が高く出たら、VLM は画像ではなく**選択肢の見た目の自然さ**で答えている。
その場合、この確認方法は事前チェックに使えない。

検証の正解率が対照より十分高ければ、VLM は画像を読んでいる。

使い方:
    .venv/bin/python experiments/m3_precheck/run_e5.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from dummies import DummyPicker, Word

VLM_ID = "Qwen/Qwen3-VL-8B-Instruct"


class Vlm:
    def __init__(self, model_id: str) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda").eval()
        self.proc = AutoProcessor.from_pretrained(model_id)

    def choose(self, image: Image.Image, options: list[str]) -> int | None:
        letters = "ABCD"[:len(options)]
        listing = "\n".join(f"{c}) {o}" for c, o in zip(letters, options))
        q = ("Look at the image. What is the main action taking place?\n"
             f"{listing}\n"
             f"Answer with a single letter ({'/'.join(letters)}) and nothing else.")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": q}]}]
        inputs = self.proc.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        text = self.proc.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip().upper()
        m = re.search(rf"\b([{letters}])\b", text)
        return letters.index(m.group(1)) if m else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m2-run", type=Path)
    p.add_argument("--model", default=VLM_ID)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = p.parse_args()

    m2_base = Path(__file__).parents[1] / "m2_japanese_roles" / "results"
    run_dir = args.m2_run or max(m2_base.glob("*/"), key=lambda d: d.name)
    m2 = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    sentences = json.loads(
        (Path(__file__).parents[1] / "m2_japanese_roles" / "sentences.json")
        .read_text(encoding="utf-8"))["sentences"]

    out_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    picker = DummyPicker()
    rng = random.Random(args.seed)
    vlm = Vlm(args.model)

    n = len(m2["records"])
    records = []
    for i, (rec, sent) in enumerate(zip(m2["records"], sentences)):
        _, _, en_predicate = sent["en"]
        _, _, ja_predicate = sent["ja"]
        match = next((w for w in picker.pool["predicates"] if w.en == en_predicate), None)
        if match is None:
            print(f"[{rec['index']:02d}] 述語 '{en_predicate}' がプールにない。飛ばす")
            continue

        question = picker.pick("predicates", Word(ja_predicate, en_predicate), rng=rng)
        options = [o["en"] for o in question["options"]]
        correct_idx = options.index(en_predicate)

        own = Image.open(run_dir / "images" / f"{rec['index']:02d}_en.png")
        # 対照: 別の文の画像。同じ選択肢を見せる(正解はこの画像に当てはまらない)
        other_i = (i + n // 2) % n
        other = Image.open(run_dir / "images" / f"{m2['records'][other_i]['index']:02d}_en.png")

        pick_own = vlm.choose(own, options)
        pick_other = vlm.choose(other, options)
        row = {"index": rec["index"], "ja_text": rec["ja_text"],
               "predicate_en": en_predicate, "options": options,
               "test_picked": options[pick_own] if pick_own is not None else None,
               "test_correct": pick_own == correct_idx,
               "control_image_index": m2["records"][other_i]["index"],
               "control_picked": options[pick_other] if pick_other is not None else None,
               "control_picked_correct_option": pick_other == correct_idx}
        records.append(row)
        print(f"[{rec['index']:02d}] {rec['ja_text']:<24} "
              f"検証={'○' if row['test_correct'] else '×'}({row['test_picked']})  "
              f"対照={'×誤通過' if row['control_picked_correct_option'] else '○'}"
              f"({row['control_picked']})", flush=True)

    total = len(records)
    test_acc = sum(r["test_correct"] for r in records) / total
    ctrl_rate = sum(r["control_picked_correct_option"] for r in records) / total
    chance = 0.25
    summary = {"n": total, "test_accuracy": round(test_acc, 3),
               "control_rate": round(ctrl_rate, 3), "chance_level": chance,
               "usable": test_acc - ctrl_rate > 0.3}
    payload = {"experiment": "E5-verb-check", "model": args.model,
               "measured_at": datetime.now().isoformat(timespec="seconds"),
               "m2_run": str(run_dir), "records": records, "summary": summary}
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== E5 まとめ({total}文) ===")
    print(f"  検証(その文の画像)での正解率 : {test_acc:.0%}")
    print(f"  対照(別の文の画像)で正解を選ぶ率: {ctrl_rate:.0%}  (当て推量は {chance:.0%})")
    print(f"  差: {test_acc - ctrl_rate:+.0%}")
    print("\n  差が大きければ、VLMは画像を読んでいる(この確認方法は使える)。")
    print("  差が小さければ、選択肢の自然さで答えているだけで、事前チェックには使えない。")
    print(f"\n書き出し: {out_dir}")


if __name__ == "__main__":
    main()
