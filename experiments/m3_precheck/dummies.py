#!/usr/bin/env python3
"""選択肢(ダミー)の生成。RULES.md §5 の実装。

4択の内訳は 正解1 + 近いダミー1 + 中くらいのダミー1 + 遠いダミー1。
「近さ」は CLIP のテキスト埋め込みのコサイン類似度で測る。
**画像照合と同じ空間を使う**ので、「CLIP空間で近い」は「絵として紛らわしい」に対応する。

RULES.md §5 の条件:

- ダミーは正解と同じカテゴリの語にする(「だれが」の候補に動詞を混ぜない)
- 正解との距離が「近すぎず遠すぎない」範囲で選ぶ。**この範囲が難易度のつまみ**
- 正解にもなりうる語(犬に対する子犬など)は避ける → 類似度の上限で弾く
- 主語と目的語の選択肢に、同じ語を並べない → 呼び出し側で除外語を渡す

使い方(選び方を目で確かめる):
    .venv/bin/python experiments/m3_precheck/dummies.py --show 犬 りんご 追いかける
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

POOL_PATH = Path(__file__).parent / "wordpool.json"

# 類似度がこれ以上の語は、正解になりうるとみなして候補から外す(犬に対する子犬など)。
# 値は暫定。E7 で難易度を調整するときに一緒に見直す【現行案】。
SYNONYM_CEILING = 0.95

# 近い/中/遠いの帯。並べ替えた候補の中の位置(0=最も近い, 1=最も遠い)で指定する。
# **ここが難易度の主要なつまみ**【現行案】。
BANDS = {"near": (0.00, 0.12), "mid": (0.40, 0.60), "far": (0.85, 1.00)}


@dataclass
class Word:
    ja: str
    en: str


class DummyPicker:
    def __init__(self, pool_path: Path = POOL_PATH, device: str = "cuda") -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.pool = {k: [Word(**w) for w in v]
                     for k, v in json.loads(pool_path.read_text(encoding="utf-8")).items()
                     if not k.startswith("_")}
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        self.embeddings = {cat: self._embed([w.en for w in words])
                           for cat, words in self.pool.items()}

    def _embed(self, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            inp = self.proc(text=texts, return_tensors="pt",
                            padding=True, truncation=True).to(self.device)
            out = self.model.get_text_features(**inp)
            # transformers 5.x は ModelOutput を返す
            f = out if torch.is_tensor(out) else out.pooler_output
            return f / f.norm(dim=-1, keepdim=True)

    def similarities(self, category: str, correct_en: str) -> list[tuple[Word, float]]:
        """候補を、正解との類似度が高い順に並べて返す(正解自身は除く)。"""
        target = self._embed([correct_en])
        sims = (self.embeddings[category] @ target.T).squeeze(1).tolist()
        pairs = [(w, s) for w, s in zip(self.pool[category], sims) if w.en != correct_en]
        return sorted(pairs, key=lambda x: -x[1])

    def pick(self, category: str, correct: Word, exclude_en: set[str] | None = None,
             rng: random.Random | None = None) -> dict:
        """正解1 + 近い1 + 中くらい1 + 遠い1 の4択を作る。"""
        rng = rng or random.Random()
        exclude = set(exclude_en or ())
        ranked = [(w, s) for w, s in self.similarities(category, correct.en)
                  if w.en not in exclude and s < SYNONYM_CEILING]
        if len(ranked) < 3:
            raise ValueError(f"候補が足りない: {category} / {correct.en}")

        chosen: list[tuple[Word, float, str]] = []
        used: set[str] = set()
        n = len(ranked)
        for band, (lo, hi) in BANDS.items():
            window = [(w, s) for w, s in ranked[int(lo * n):max(int(hi * n), int(lo * n) + 1)]
                      if w.en not in used]
            if not window:  # 帯が空なら、未使用の中から最も近いものを繰り上げる
                window = [(w, s) for w, s in ranked if w.en not in used]
            w, s = rng.choice(window)
            used.add(w.en)
            chosen.append((w, s, band))

        options = [{"ja": correct.ja, "en": correct.en, "role": "correct", "sim": 1.0}]
        options += [{"ja": w.ja, "en": w.en, "role": band, "sim": round(s, 4)}
                    for w, s, band in chosen]
        rng.shuffle(options)   # 正解の位置を散らす
        return {"correct_en": correct.en, "options": options}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--show", nargs=3, metavar=("SUBJECT", "OBJECT", "PREDICATE"),
                   default=["犬", "りんご", "追いかける"],
                   help="この3語について4択を作って表示する(日本語で指定)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    picker = DummyPicker()
    rng = random.Random(args.seed)
    cats = [("subjects", "だれが"), ("objects", "なにを"), ("predicates", "どうした")]

    used_en: set[str] = set()
    for (cat, label), ja in zip(cats, args.show):
        match = next((w for w in picker.pool[cat] if w.ja == ja), None)
        if match is None:
            print(f"{label}: 「{ja}」はプールにない")
            continue
        # 主語と目的語に同じ語を並べない(RULES.md §5)
        result = picker.pick(cat, match, exclude_en=used_en, rng=rng)
        used_en |= {o["en"] for o in result["options"]}
        print(f"\n{label}(正解: {ja})")
        for o in result["options"]:
            mark = "★" if o["role"] == "correct" else "  "
            print(f"  {mark} {o['ja']:<12} {o['en']:<24} {o['role']:<8} sim={o['sim']}")


if __name__ == "__main__":
    main()
