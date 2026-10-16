#!/usr/bin/env python3
"""M2: E2(主語と目的語の役割の再現)と E4(日本語の扱い)を実測する。

計画書 §6 の E2・E4。知りたいのは2点。

- **E2**: 「犬が猫を追いかける」を、猫が犬を追う絵に描いてしまわないか。
  崩れるなら、入れ替えた回答も正解として認める必要がある(計画書 §8-14、RULES.md §11)。
- **E4**: 日本語をそのまま渡して通るか、英訳が要るか。直入力が使えれば、
  1ターンあたりの翻訳(LLM1回)の待ち時間を省ける。

生成するのは3種類。

| 種別 | 内容 | 使う検証 |
|---|---|---|
| ja | 日本語の文をそのまま渡す | E4 |
| en | 英訳を渡す | E2・E4 |
| en_swapped | 主語と目的語を入れ替えた英文 | E2 |

**E2 の集計は `reversible: true` の組だけで行う**。入れ替えて意味が壊れる組
(りんごが男の子を食べる等)を混ぜると、役割の取り違えなのか、そもそも描けないのかを
切り分けられなくなるため。

判定は別スクリプト(VLM)に任せ、ここでは生成と、CLIPによる補助的な数値だけを出す。
**CLIP は語順・役割の判別が苦手**なので、E2 の判定には使えない(補助として残すだけ)。

使い方:
    .venv/bin/python experiments/m2_japanese_roles/run_m2.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

# PIL の既定フォントは日本語を豆腐文字にするので、CJK フォントを探して使う。
CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
]


def label_font(size: int = 14) -> ImageFont.ImageFont:
    for path in CJK_FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def ja_sentence(subject: str, obj: str, predicate: str) -> str:
    """ゲームと同じ組み立て方。だれが + が + なにを + を + どうした。"""
    return f"{subject}が{obj}を{predicate}"


def en_sentence(subject: str, obj: str, predicate: str) -> str:
    return f"{subject} {predicate} {obj}"


def load_pipeline(model_id: str):
    from diffusers import ZImagePipeline

    try:
        pipe = ZImagePipeline.from_pretrained(model_id, dtype=torch.bfloat16, low_cpu_mem_usage=False)
    except TypeError:
        pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    return pipe.to("cuda")


class Clip:
    """CLIP の text-image スコア。**役割の判定には使えない**ので補助扱い。"""

    def __init__(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @staticmethod
    def _tensor(out):
        # transformers 5.x は ModelOutput を返す。射影済みは pooler_output。
        return out if torch.is_tensor(out) else out.pooler_output

    def scores(self, image: Image.Image, texts: list[str]) -> list[float]:
        with torch.no_grad():
            ii = self.proc(images=image, return_tensors="pt").to("cuda")
            ti = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            f_i = self._tensor(self.model.get_image_features(**ii))
            f_t = self._tensor(self.model.get_text_features(**ti))
            f_i = f_i / f_i.norm(dim=-1, keepdim=True)
            f_t = f_t / f_t.norm(dim=-1, keepdim=True)
            return (f_i @ f_t.T)[0].tolist()


def sheet(images: list[Image.Image], labels: list[str], cell: int = 320) -> Image.Image:
    font = label_font()
    img = Image.new("RGB", (cell * len(images), cell + 22), "black")
    draw = ImageDraw.Draw(img)
    for i, (im, label) in enumerate(zip(images, labels)):
        img.paste(im.resize((cell, cell)), (cell * i, 22))
        draw.text((cell * i + 4, 4), label, fill="white", font=font)
    return img


def rebuild_sheets(run_dir: Path, sentences: list[dict]) -> None:
    """保存済みの画像からシートだけ作り直す。GPU も生成も不要。"""
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    for rec in data["records"]:
        images, labels = [], []
        for f in rec["files"]:
            kind = f.split("_", 1)[1].removesuffix(".png")
            prompt = {"ja": rec["ja_text"], "en": rec["en_text"],
                      "en_swapped": rec["en_swapped_text"]}[kind]
            images.append(Image.open(run_dir / "images" / f))
            labels.append(f"{kind}: {prompt}")
        sheet(images, labels).save(run_dir / f"sheet_{rec['index']:02d}.png")
    print(f"シートを作り直した: {run_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sentences", type=Path,
                   default=Path(__file__).parent / "sentences.json")
    p.add_argument("--size", type=int, default=512, help="512で確定済み")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    p.add_argument("--sheets-only", type=Path, metavar="RUN_DIR",
                   help="既存の結果からシートだけ作り直す(GPU不要)")
    args = p.parse_args()

    data = json.loads(args.sentences.read_text(encoding="utf-8"))["sentences"]
    if args.sheets_only:
        rebuild_sheets(args.sheets_only, data)
        return
    out_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    print(f"文の数: {len(data)}(うち reversible {sum(d['reversible'] for d in data)})")
    pipe = load_pipeline(MODEL_ID)
    clip = Clip()

    def generate(prompt: str, seed: int) -> Image.Image:
        return pipe(prompt=prompt, height=args.size, width=args.size,
                    num_inference_steps=args.steps, guidance_scale=0.0,
                    generator=torch.Generator("cuda").manual_seed(seed)).images[0]

    records = []
    for idx, d in enumerate(data):
        ja_s, ja_o, ja_v = d["ja"]
        en_s, en_o, en_v = d["en"]
        ja_text = ja_sentence(ja_s, ja_o, ja_v)
        en_text = en_sentence(en_s, en_o, en_v)
        en_swapped_text = en_sentence(en_o, en_s, en_v)
        seed = args.seed + idx * 100

        variants = {"ja": ja_text, "en": en_text}
        if d["reversible"]:
            variants["en_swapped"] = en_swapped_text

        rec = {"index": idx, "ja_text": ja_text, "en_text": en_text,
               "en_swapped_text": en_swapped_text, "reversible": d["reversible"],
               "homonym": d.get("homonym"), "seed": seed, "clip": {}}

        images, labels = [], []
        for kind, prompt in variants.items():
            img = generate(prompt, seed)
            name = f"{idx:02d}_{kind}.png"
            img.save(out_dir / "images" / name)
            images.append(img)
            labels.append(f"{kind}: {prompt}")
            # 補助の数値。正順と逆順のどちらに近いか(CLIPは語順に弱いので参考値)。
            s_normal, s_swapped = clip.scores(img, [en_text, en_swapped_text])
            rec["clip"][kind] = {"to_normal": round(s_normal, 4),
                                 "to_swapped": round(s_swapped, 4),
                                 "margin": round(s_normal - s_swapped, 4)}
        rec["files"] = [f"{idx:02d}_{k}.png" for k in variants]
        sheet(images, labels).save(out_dir / f"sheet_{idx:02d}.png")
        records.append(rec)
        print(f"[{idx:02d}] {ja_text} / {en_text}"
              + ("  (reversible)" if d["reversible"] else ""), flush=True)

    payload = {"experiment": "M2-japanese-and-roles",
               "measured_at": datetime.now().isoformat(timespec="seconds"),
               "model_id": MODEL_ID, "size": args.size, "steps": args.steps,
               "records": records}
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n書き出し: {out_dir}")
    print("判定は judge_vlm.py で行う。CLIP の margin は語順に弱いため参考値。")


if __name__ == "__main__":
    main()
