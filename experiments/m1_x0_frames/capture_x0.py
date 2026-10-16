#!/usr/bin/env python3
"""M1 / E1(後半): 各ステップの x0 予測を保存し、最終画像との類似度曲線を作る。

計画書 §3.2 の手順1〜3に対応する。知りたいのは次の3点。

1. **x0 予測が段階的に鮮明になるか**。Z-Image-Turbo は DMD 系の蒸留モデルで、
   序盤のステップからすでに鮮明な画像が出る可能性がある(計画書 §3.2 の注意点)。
   そうなら「ノイズから浮かび上がる」というゲームの核が成立しないので、
   線形補間(§3.3)か Z-Image-Base への切り替えを検討することになる。
2. **見せるフレームをどう選ぶか**。類似度曲線を等間隔に区切り、各区切りに最も近い
   ステップを採る(§3.2 手順3)。序盤の色の塊と終盤のほぼ同じ画像が間引かれる。
3. **文ごとに選び直すか、全体で固定するか**(計画書 §8-4)。複数の文で曲線を取り、
   選ばれるステップがばらつくかを見る。テンポの安定のため固定が扱いやすい見込み。

## x0 予測の取り出し方

diffusers の `ZImagePipeline` は、フローマッチングの Euler 法で
`x_{i+1} = x_i + (σ_{i+1} - σ_i)·v_i` と進む(`noise_pred` が速度場 v)。
x0 予測は同じ直線上にあるので、

    x0_i = x_i - σ_i·v_i = x_{i+1} - σ_{i+1}·v_i

後者を使えば、コールバックが受け取る「ステップ後の latents」と `noise_pred` だけで
厳密に復元できる。最終ステップは σ=0 なので x0 は最終 latents と一致し、
これが復元の検算になる(--verify で確認する)。

**途中の潜在変数をそのままデコードしてはいけない**(序盤がほぼノイズになる)。

使い方:
    .venv/bin/python experiments/m1_x0_frames/capture_x0.py
    .venv/bin/python experiments/m1_x0_frames/capture_x0.py --steps 12 --frames 7
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

# ゲームで流れるのと同じ「だれが・なにを・どうした」の3語文。主語と目的語の種類、
# 動詞の描きやすさがばらけるように選ぶ(計画書 §3.5: 英訳を前提)。
DEFAULT_PROMPTS = [
    "A dog chases a cat",
    "A boy eats an apple",
    "A cat pushes a cup",
    "A woman reads a book",
    "An elephant carries a log",
    "A robot waters a plant",
]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_pipeline(model_id: str):
    from diffusers import ZImagePipeline

    try:
        pipe = ZImagePipeline.from_pretrained(model_id, dtype=torch.bfloat16, low_cpu_mem_usage=False)
    except TypeError:
        pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    return pipe.to("cuda")


def generate_with_x0(pipe, prompt: str, size: int, steps: int, seed: int):
    """1文を生成し、各ステップの x0 予測(latent)と最終画像を返す。"""
    x0_latents: list[torch.Tensor] = []

    def callback(pipe_, i, t, kwargs):
        # kwargs["latents"] はステップ後の x_{i+1}、kwargs["noise_pred"] が速度場 v_i。
        sigma_next = pipe_.scheduler.sigmas[i + 1].to(torch.float32)
        assert pipe_.scheduler.step_index == i + 1, "sigma の添字がステップとずれている"
        x0 = kwargs["latents"].to(torch.float32) - sigma_next * kwargs["noise_pred"].to(torch.float32)
        x0_latents.append(x0.detach().cpu())
        return {}

    # noise_pred はパイプラインのローカル変数にあるが、既定では取り出しが許可されていない。
    # クラス側のリストを壊さないよう、インスタンス属性で上書きする。
    pipe._callback_tensor_inputs = list(type(pipe)._callback_tensor_inputs) + ["noise_pred"]

    out = pipe(
        prompt=prompt,
        height=size,
        width=size,
        num_inference_steps=steps,
        guidance_scale=0.0,  # Turbo は CFG 不要
        generator=torch.Generator("cuda").manual_seed(seed),
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents", "noise_pred"],
    )
    return x0_latents, out.images[0]


def decode_latent(pipe, latent: torch.Tensor) -> Image.Image:
    """x0 の latent を画像に戻す。パイプライン末尾と同じ手順。"""
    lat = latent.to("cuda", pipe.vae.dtype)
    lat = lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    with torch.no_grad():
        img = pipe.vae.decode(lat, return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type="pil")[0]


class Metrics:
    """最終画像との隔たりを2つの尺度で測る。

    - LPIPS: 見た目の違い。「まだぼやけているか」に近い。
    - CLIP類似度: 意味の違い。「何が写っているか分かるか」に近い。
      人がいつ語を判別できるかは後者に近いが、どちらも代用品でしかない(計画書 §6 の注記)。
    """

    def __init__(self) -> None:
        import lpips
        from transformers import CLIPModel, CLIPProcessor

        self.lpips = lpips.LPIPS(net="alex").to("cuda").eval()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        import numpy as np

        a = torch.from_numpy(np.asarray(img.convert("RGB"), dtype="float32") / 255.0)
        return (a.permute(2, 0, 1).unsqueeze(0) * 2 - 1).to("cuda")  # [-1,1]

    def lpips_distance(self, a: Image.Image, b: Image.Image) -> float:
        with torch.no_grad():
            return float(self.lpips(self._to_tensor(a), self._to_tensor(b)).item())

    def clip_similarity(self, a: Image.Image, b: Image.Image) -> float:
        with torch.no_grad():
            inp = self.clip_proc(images=[a, b], return_tensors="pt").to("cuda")
            out = self.clip.get_image_features(**inp)
            # transformers 5.x では ModelOutput が返り、射影済みの埋め込みは pooler_output に入る。
            f = out if torch.is_tensor(out) else out.pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
            return float((f[0] @ f[1]).item())


def pick_frames(progress: list[float], n_frames: int) -> list[int]:
    """知覚上の進み具合 progress(0=ノイズ, 1=最終画像)を等間隔に区切って選ぶ。

    計画書 §3.2 手順3。単純にステップを等間隔で採ると、序盤の似た色の塊と
    終盤のほぼ同じ画像ばかりになるため、曲線の側を等分する。
    最後のフレームは必ず最終画像(最後のステップ)にする。
    """
    targets = [i / (n_frames - 1) for i in range(n_frames)]
    picked: list[int] = []
    for target in targets:
        idx = min(range(len(progress)), key=lambda k: abs(progress[k] - target))
        picked.append(idx)
    picked[-1] = len(progress) - 1
    # 同じステップが重複したら、後ろにずらして重複を避ける
    seen: list[int] = []
    for idx in picked:
        while idx in seen and idx + 1 < len(progress):
            idx += 1
        seen.append(idx)
    return seen


def contact_sheet(frames: list[Image.Image], labels: list[str], cell: int = 256) -> Image.Image:
    """フレームを横1列に並べ、ステップ番号を焼き込んだ確認用の画像。"""
    sheet = Image.new("RGB", (cell * len(frames), cell + 18), "black")
    for i, (img, label) in enumerate(zip(frames, labels)):
        sheet.paste(img.resize((cell, cell)), (cell * i, 18))
        ImageDraw.Draw(sheet).text((cell * i + 4, 4), label, fill="white")
    return sheet


def interpolation_series(final: Image.Image, alphas: list[float], seed: int) -> list[Image.Image]:
    """E6 用の別案: 最終画像と固定ノイズの線形補間(画素空間)。

    x0 予測が「ぼやけた状態から鮮明に」なるのに対し、こちらは
    「粒状のノイズ越しに構図が透ける」見え方になる(計画書 §3.3)。
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(final.height, final.width, 3), dtype="uint8")
    base = np.asarray(final.convert("RGB"), dtype="float32")
    out = []
    for a in alphas:
        blended = base * a + noise.astype("float32") * (1 - a)
        out.append(Image.fromarray(blended.clip(0, 255).astype("uint8")))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    p.add_argument("--size", type=int, default=512, help="解像度。512で確定済み")
    p.add_argument("--steps", type=int, default=8, help="NFE と同数の x0 予測が取れる")
    p.add_argument("--frames", type=int, default=7, help="見せるフレーム枚数(6〜7が目安)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-interpolation", action="store_true", help="E6用の線形補間を作らない")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA が使えない。")

    out_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    print(f"読み込み中: {args.model}")
    pipe = load_pipeline(args.model)
    metrics = Metrics()

    records = []
    for prompt in args.prompts:
        slug = slugify(prompt)
        print(f"\n[{prompt}]", flush=True)
        x0_latents, final_image = generate_with_x0(pipe, prompt, args.size, args.steps, args.seed)
        assert len(x0_latents) == args.steps, f"x0 が {len(x0_latents)} 枚しか取れていない"

        frame_dir = out_dir / "frames" / slug
        frame_dir.mkdir(parents=True, exist_ok=True)
        images = []
        for i, lat in enumerate(x0_latents):
            img = decode_latent(pipe, lat)
            img.save(frame_dir / f"step{i:02d}.png")
            images.append(img)
        final_image.save(frame_dir / "final.png")

        # 検算: 最後の x0 は σ=0 なので最終画像と一致するはず。
        import numpy as np
        max_diff = int(np.abs(np.asarray(images[-1], dtype="int16")
                              - np.asarray(final_image, dtype="int16")).max())
        print(f"  検算(最終x0 vs 最終画像の画素差の最大): {max_diff}")

        lpips_curve = [metrics.lpips_distance(img, final_image) for img in images]
        clip_curve = [metrics.clip_similarity(img, final_image) for img in images]
        # 知覚上の進み具合。0 = 1枚目と同じ隔たり、1 = 最終画像。
        l0 = lpips_curve[0] or 1e-8
        progress = [1 - (d / l0) for d in lpips_curve]
        picked = pick_frames(progress, args.frames)

        print("  step : LPIPS  CLIP   progress")
        for i, (d, c, pr) in enumerate(zip(lpips_curve, clip_curve, progress)):
            mark = " <-" if i in picked else ""
            print(f"   {i:2d}  : {d:.3f}  {c:.3f}  {pr:+.2f}{mark}")
        print(f"  選んだステップ: {picked}")

        contact_sheet(images, [f"step{i}" for i in range(len(images))]).save(
            out_dir / f"sheet_all_{slug}.png")
        contact_sheet([images[i] for i in picked], [f"step{i}" for i in picked]).save(
            out_dir / f"sheet_picked_{slug}.png")

        if not args.no_interpolation:
            alphas = [i / (args.frames - 1) for i in range(args.frames)]
            series = interpolation_series(final_image, alphas, args.seed)
            contact_sheet(series, [f"a={a:.2f}" for a in alphas]).save(
                out_dir / f"sheet_interp_{slug}.png")

        records.append({
            "prompt": prompt,
            "slug": slug,
            "lpips_to_final": [round(v, 4) for v in lpips_curve],
            "clip_sim_to_final": [round(v, 4) for v in clip_curve],
            "progress": [round(v, 4) for v in progress],
            "picked_steps": picked,
            "final_x0_max_pixel_diff": max_diff,
        })

    # 文ごとに選ばれ方がばらつくか(計画書 §8-4)。
    picks = [tuple(r["picked_steps"]) for r in records]
    agreement = {"identical_across_prompts": len(set(picks)) == 1,
                 "distinct_selections": [list(x) for x in sorted(set(picks))]}
    print(f"\n選び方が全文で一致したか: {agreement['identical_across_prompts']}")

    payload = {
        "experiment": "M1-x0-frames",
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": args.model, "size": args.size, "steps": args.steps,
        "frames": args.frames, "seed": args.seed,
        "records": records, "frame_selection_agreement": agreement,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for r in records:
            axes[0].plot(r["lpips_to_final"], marker="o", label=r["prompt"])
            axes[1].plot(r["clip_sim_to_final"], marker="o", label=r["prompt"])
        axes[0].set_title("LPIPS to final (lower = closer)")
        axes[1].set_title("CLIP image similarity to final")
        for ax in axes:
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "curves.png", dpi=120)
    except ImportError:
        print("matplotlib が無いのでグラフは省略")

    print(f"\n書き出し: {out_dir}")


if __name__ == "__main__":
    main()
