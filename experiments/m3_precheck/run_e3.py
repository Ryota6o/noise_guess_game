#!/usr/bin/env python3
"""E3: 3語の判別と、事前チェックの通過率。計画書 §6 の E3、§3.4 の手順1。

M2 が 512px で生成した画像を再利用する(解像度は512で確定済み)。

## 何を測るか

1. **4択での正解率**。事前チェックの候補集合は、**実際にプレイヤーに出す4択と同じ**にする。
   CLIP が4択から正解を選べないなら、プレイヤーにも選べない見込みが高い。
   これが事前チェックの本番に最も近い形。
2. **フルプール(約50語)での順位**。語そのものの判別しやすさを見る。4択より厳しい。
3. **マージン**(正解スコア − 2位スコア)の分布。**閾値はここから決める**。

## 「入力された語そのもの」で判定する

ウミヘビの試し(2026-10-16)で分かったこと。「ヒロオウミヘビ」と入力して出た絵は、
ただのヘビだった。この絵を「ヘビ」で照合すれば通ってしまうが、**プレイヤーが選ぶのは
「ヒロオウミヘビ」**なので、通してはいけない。候補集合は必ず**入力語**を含む形で作る。

使い方:
    .venv/bin/python experiments/m3_precheck/run_e3.py
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from dummies import DummyPicker, Word


def clip_rank(picker: DummyPicker, image: Image.Image, candidates: list[str]) -> dict:
    """候補語の中で、画像に最も近いものを選ぶ(CLIPゼロショット分類)。"""
    with torch.no_grad():
        ii = picker.proc(images=image, return_tensors="pt").to(picker.device)
        out = picker.model.get_image_features(**ii)
        f_i = out if torch.is_tensor(out) else out.pooler_output
        f_i = f_i / f_i.norm(dim=-1, keepdim=True)
    f_t = picker._embed(candidates)
    sims = (f_i @ f_t.T)[0].tolist()
    order = sorted(range(len(candidates)), key=lambda i: -sims[i])
    return {"scores": {candidates[i]: round(sims[i], 4) for i in order},
            "ranking": [candidates[i] for i in order]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m2-run", type=Path,
                   help="run_m2.py の出力。省略時は最新")
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
    pool_s = [w.en for w in picker.pool["subjects"]]
    pool_o = [w.en for w in picker.pool["objects"]]

    records = []
    for rec, sent in zip(m2["records"], sentences):
        en_subject, en_object, _ = sent["en"]
        ja_subject, ja_object, _ = sent["ja"]
        # 英訳で確定済み(E4)なので en 画像で判定する
        image = Image.open(run_dir / "images" / f"{rec['index']:02d}_en.png")

        # 実際に出す4択を作る。主語と目的語に同じ語を並べない(RULES.md §5)
        sub_q = picker.pick("subjects", Word(ja_subject, en_subject), rng=rng)
        obj_q = picker.pick("objects", Word(ja_object, en_object),
                            exclude_en={o["en"] for o in sub_q["options"]}, rng=rng)

        row = {"index": rec["index"], "ja_text": rec["ja_text"], "en_text": rec["en_text"],
               "subject_en": en_subject, "object_en": en_object}
        for slot, question, pool, correct in (
                ("subject", sub_q, pool_s, en_subject),
                ("object", obj_q, pool_o, en_object)):
            four = [o["en"] for o in question["options"]]
            r4 = clip_rank(picker, image, four)
            rfull = clip_rank(picker, image, pool)
            top4 = r4["ranking"][0]
            scores = list(r4["scores"].values())
            row[slot] = {
                "options": question["options"],
                "four_choice_top1": top4,
                "four_choice_correct": top4 == correct,
                "four_choice_margin": round(scores[0] - scores[1], 4),
                "correct_score": r4["scores"][correct],
                "full_pool_rank": rfull["ranking"].index(correct) + 1,
                "full_pool_top1": rfull["ranking"][0],
            }
        records.append(row)
        print(f"[{rec['index']:02d}] {rec['ja_text']:<24} "
              f"主語 4択={'○' if row['subject']['four_choice_correct'] else '×'}"
              f"(順位{row['subject']['full_pool_rank']:>2})  "
              f"目的語 4択={'○' if row['object']['four_choice_correct'] else '×'}"
              f"(順位{row['object']['full_pool_rank']:>2})", flush=True)

    def stats(slot: str) -> dict:
        vals = [r[slot] for r in records]
        ok = [v for v in vals if v["four_choice_correct"]]
        margins_ok = sorted(v["four_choice_margin"] for v in ok)
        margins_ng = sorted(v["four_choice_margin"] for v in vals
                            if not v["four_choice_correct"])
        return {
            "four_choice_accuracy": round(len(ok) / len(vals), 3),
            "full_pool_top1_rate": round(
                sum(1 for v in vals if v["full_pool_rank"] == 1) / len(vals), 3),
            "full_pool_top3_rate": round(
                sum(1 for v in vals if v["full_pool_rank"] <= 3) / len(vals), 3),
            "margin_when_correct": {
                "min": margins_ok[0] if margins_ok else None,
                "median": margins_ok[len(margins_ok) // 2] if margins_ok else None,
                "max": margins_ok[-1] if margins_ok else None},
            "margin_when_wrong": {
                "min": margins_ng[0] if margins_ng else None,
                "median": margins_ng[len(margins_ng) // 2] if margins_ng else None,
                "max": margins_ng[-1] if margins_ng else None},
        }

    summary = {slot: stats(slot) for slot in ("subject", "object")}
    payload = {"experiment": "E3-word-discriminability",
               "measured_at": datetime.now().isoformat(timespec="seconds"),
               "m2_run": str(run_dir), "size": m2["size"],
               "records": records, "summary": summary}
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== E3 まとめ(512px、28文) ===")
    for slot, s in summary.items():
        name = {"subject": "主語", "object": "目的語"}[slot]
        print(f"\n{name}")
        print(f"  4択での正解率     : {s['four_choice_accuracy']:.0%}")
        print(f"  フルプールで1位   : {s['full_pool_top1_rate']:.0%}  "
              f"(3位以内 {s['full_pool_top3_rate']:.0%})")
        print(f"  マージン(正解時) : 最小 {s['margin_when_correct']['min']} / "
              f"中央 {s['margin_when_correct']['median']}")
        print(f"  マージン(誤り時) : 中央 {s['margin_when_wrong']['median']} / "
              f"最大 {s['margin_when_wrong']['max']}")
    print(f"\n書き出し: {out_dir}")


if __name__ == "__main__":
    main()
