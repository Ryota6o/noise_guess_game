#!/usr/bin/env python3
"""E4 の追加計測: 日本語プロンプトが「文字を描け」と解釈されていないか。

M2 の画像を見ると、日本語を直接渡した場合に、場面ではなく**日本語の文字が画像に
描き込まれる**ことがある(例: 「警察官が泥棒を捕まえる」で崩れた日本語が大きく描かれる)。
Z-Image は文字描画(英語・中国語)を売りにしたモデルなので、日本語の文字列を
「描くべき文字」と解釈しうる。

出現率(主語・目的語が写っているか)とは別の失敗モードなので、独立に数える。
「文字が写っているか」は常識の入り込む余地がない問いなので、VLM判定でも偏りにくい。

使い方:
    .venv/bin/python experiments/m2_japanese_roles/probe_text_artifacts.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

VLM_ID = "Qwen/Qwen3-VL-8B-Instruct"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path)
    p.add_argument("--model", default=VLM_ID)
    args = p.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    base = Path(__file__).parent / "results"
    run_dir = args.run or max(base.glob("*/"), key=lambda d: d.name)
    data = json.loads((run_dir / "results_judged.json").read_text(encoding="utf-8"))

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(args.model)

    def ask(image: Image.Image, question: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": question}]}]
        inputs = proc.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        return proc.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip().lower()

    q = ("Does this image contain written words, letters or characters as a visible "
         "part of the picture (for example text painted across the image)? "
         "Ignore small incidental signs in the background. "
         "Answer with a single word, yes or no.")

    counts = {"ja": 0, "en": 0}
    totals = {"ja": 0, "en": 0}
    for rec in data["records"]:
        i = rec["index"]
        rec["text_artifact"] = {}
        for kind in ("ja", "en"):
            path = run_dir / "images" / f"{i:02d}_{kind}.png"
            if not path.exists():
                continue
            ans = ask(Image.open(path), q)
            has = "yes" in ans
            rec["text_artifact"][kind] = has
            totals[kind] += 1
            counts[kind] += int(has)
        if rec["text_artifact"].get("ja") and not rec["text_artifact"].get("en"):
            print(f"[{i:02d}] {rec['ja_text']}  <- 日本語版だけ文字が出た")

    data["text_artifact_summary"] = {
        k: {"with_text": counts[k], "total": totals[k],
            "rate": round(counts[k] / totals[k], 3) if totals[k] else 0.0}
        for k in ("ja", "en")}
    (run_dir / "results_judged.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 画像に文字が描き込まれた割合 ===")
    for k, v in data["text_artifact_summary"].items():
        print(f"  {k}: {v['with_text']}/{v['total']} = {v['rate']:.0%}")


if __name__ == "__main__":
    main()
