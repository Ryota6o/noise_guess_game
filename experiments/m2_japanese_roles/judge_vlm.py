#!/usr/bin/env python3
"""M2 の生成画像を VLM で判定する(E2・E4)。

`run_m2.py` が出した画像を読み、Qwen3-VL に質問して集計する。
CLIP は語順・役割の判別が苦手なので、ここでは使わない(計画書 §3.4、§7)。

## E2: 役割の再現をどう測るか

「犬が猫を追いかける」の絵と「猫が犬を追いかける」の絵を別々に作り、
それぞれについて VLM に2択で聞く。**2枚の答えが食い違ったかどうか**を見るのが要点。

| 2枚の判定 | 意味 |
|---|---|
| 食い違い、かつ正しい向き | 生成器が役割を描き分けている(対策不要) |
| 食い違い、だが逆向き | 役割が一貫して反転している |
| 一致(同じ答え) | **生成器が役割を描き分けていない**。入れ替えた回答も正解にする必要がある |

片方の画像だけを見て正解率を出すと、VLM 自身の誤りと生成器の誤りが混ざる。
2枚の差を見れば、VLM に共通の偏りがあっても相殺される。

選択肢の順番は画像ごとにランダムに入れ替える(VLM が常に A を選ぶ偏りを打ち消すため)。

## E4: 日本語直入力の可否

同じ文を日本語そのままと英訳で生成し、**主語と目的語が絵に出ているか**を VLM に聞いて、
出現率を比べる。日本語直入力が英訳に見劣りしなければ、翻訳工程(LLM1回)を省ける。

使い方:
    .venv/bin/python experiments/m2_japanese_roles/judge_vlm.py          # 最新の結果を判定
    .venv/bin/python experiments/m2_japanese_roles/judge_vlm.py --run <dir>
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
    def __init__(self, model_id: str = VLM_ID) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda").eval()
        self.proc = AutoProcessor.from_pretrained(model_id)

    def ask(self, image: Image.Image, question: str, max_new_tokens: int = 24) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        inputs = self.proc.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        text = self.proc.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        return text.strip()

    def choose(self, image: Image.Image, option_a: str, option_b: str) -> str | None:
        """2択。返り値は 'a' / 'b' / None(判定不能)。"""
        q = (f"Look carefully at the image. Which sentence describes what is happening?\n"
             f"A) {option_a}\nB) {option_b}\n"
             f"Answer with a single letter, A or B, and nothing else.")
        ans = self.ask(image, q, max_new_tokens=8).upper()
        m = re.search(r"\b([AB])\b", ans)
        return m.group(1).lower() if m else None

    def yes_no(self, image: Image.Image, thing: str) -> bool | None:
        q = (f"Does this image clearly show {thing}? "
             f"Answer with a single word, yes or no, and nothing else.")
        ans = self.ask(image, q, max_new_tokens=8).lower()
        if "yes" in ans:
            return True
        if "no" in ans:
            return False
        return None


def judge_roles(vlm: Vlm, images: dict[str, Image.Image], normal: str, swapped: str,
                rng: random.Random) -> dict:
    """正順の画像と逆順の画像で、VLM の答えが食い違うかを見る。"""
    result = {}
    for kind in ("en", "en_swapped"):
        if kind not in images:
            continue
        # 選択肢の並び順をランダムにして、位置の偏りを打ち消す
        normal_first = rng.random() < 0.5
        a, b = (normal, swapped) if normal_first else (swapped, normal)
        letter = vlm.choose(images[kind], a, b)
        if letter is None:
            picked = None
        else:
            picked = ("normal" if normal_first else "swapped") if letter == "a" \
                else ("swapped" if normal_first else "normal")
        result[kind] = {"picked": picked, "normal_shown_first": normal_first}

    pick_en = result.get("en", {}).get("picked")
    pick_sw = result.get("en_swapped", {}).get("picked")
    if pick_en is None or pick_sw is None:
        verdict = "undetermined"
    elif pick_en == pick_sw:
        verdict = "not_distinguished"          # 生成器が役割を描き分けていない
    elif pick_en == "normal" and pick_sw == "swapped":
        verdict = "distinguished_correct"      # 描き分けている
    else:
        verdict = "distinguished_reversed"     # 一貫して反転している
    result["verdict"] = verdict
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, help="run_m2.py の出力ディレクトリ。省略時は最新")
    p.add_argument("--sentences", type=Path,
                   default=Path(__file__).parent / "sentences.json",
                   help="主語・目的語を個別に聞くために参照する")
    p.add_argument("--model", default=VLM_ID)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    base = Path(__file__).parent / "results"
    run_dir = args.run or max(base.glob("*/"), key=lambda d: d.name)
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    sentences = json.loads(args.sentences.read_text(encoding="utf-8"))["sentences"]
    assert len(sentences) == len(data["records"]), "文リストと結果の数が合わない"
    print(f"判定対象: {run_dir}  ({len(data['records'])} 文)")

    vlm = Vlm(args.model)
    rng = random.Random(args.seed)

    for rec in data["records"]:
        images = {}
        for f in rec["files"]:
            kind = f.split("_", 1)[1].removesuffix(".png")
            images[kind] = Image.open(run_dir / "images" / f)

        # E4: 主語・目的語が絵に出ているか(日本語直入力 vs 英訳)
        en_subject, en_object, _ = sentences[rec["index"]]["en"]
        rec["vlm_presence"] = {}
        for kind in ("ja", "en"):
            if kind not in images:
                continue
            rec["vlm_presence"][kind] = {
                "subject": vlm.yes_no(images[kind], en_subject),
                "object": vlm.yes_no(images[kind], en_object),
                "whole": vlm.yes_no(images[kind], rec["en_text"]),
                "caption": vlm.ask(images[kind], "Describe this image in one short sentence.", 40),
            }

        # E2: 役割の描き分け(reversible の組だけ)
        if rec["reversible"]:
            rec["vlm_roles"] = judge_roles(vlm, images, rec["en_text"],
                                           rec["en_swapped_text"], rng)
            print(f"[{rec['index']:02d}] {rec['ja_text']}  -> {rec['vlm_roles']['verdict']}")
        else:
            pres = rec["vlm_presence"]
            print(f"[{rec['index']:02d}] {rec['ja_text']}  "
                  f"ja(S/O)={pres['ja']['subject']}/{pres['ja']['object']}  "
                  f"en(S/O)={pres['en']['subject']}/{pres['en']['object']}")

    # 集計
    rev = [r for r in data["records"] if r["reversible"]]
    verdicts: dict[str, int] = {}
    for r in rev:
        v = r["vlm_roles"]["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    def rate(kind: str, field: str) -> float:
        vals = [r["vlm_presence"][kind][field] for r in data["records"]
                if kind in r["vlm_presence"]]
        return sum(1 for v in vals if v) / len(vals) if vals else 0.0

    summary = {
        "E2_role_verdicts": verdicts,
        "E2_reversible_count": len(rev),
        "E4_presence_rate": {
            kind: {field: round(rate(kind, field), 3)
                   for field in ("subject", "object", "whole")}
            for kind in ("ja", "en")
        },
    }
    data["vlm_summary"] = summary
    data["vlm_model"] = args.model
    (run_dir / "results_judged.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== E2: 役割の描き分け(reversible のみ) ===")
    for k, v in sorted(verdicts.items()):
        print(f"  {k}: {v}/{len(rev)}")
    print("\n=== E4: 主語・目的語の出現率 ===")
    for kind, vals in summary["E4_presence_rate"].items():
        print(f"  {kind}: 主語 {vals['subject']:.0%} / 目的語 {vals['object']:.0%} / 文全体 {vals['whole']:.0%}")
    print(f"\n書き出し: {run_dir / 'results_judged.json'}")


if __name__ == "__main__":
    main()
