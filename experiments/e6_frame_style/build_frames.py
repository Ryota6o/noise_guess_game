#!/usr/bin/env python3
"""線形補間でフレーム列を作り、§3.3 のタイミングで並べて確認する。

計画書 §3.2(確定: 線形補間)の実装と、その見え方の確認。
M1 で x0 予測の方式が否定されたため、これがフレームの作り方の本体になる。

    frame(α) = final × α + noise × (1-α)
    α(i) = α0 + (1 - α0) × (i / (n-1))^γ

γ が難易度の主要なつまみ。大きいほど鮮明になるのが遅れ、難しくなる。
確定はE7(正答までの時間の分布)の実測後。

使い方(M1 が生成した最終画像を再利用する):
    .venv/bin/python experiments/e6_frame_style/build_frames.py \
        --finals experiments/m1_x0_frames/results/*/frames/*/final.png
    .venv/bin/python experiments/e6_frame_style/build_frames.py --gammas 1.0 2.0 3.0
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# §3.3: 0〜7秒を1秒刻みで8フレーム。切り替えが5.0秒と6.0秒になるので、
# 5.5秒のヒントが両隣から最も離れる。
FRAME_TIMES_S = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
HINT_TIME_S = 5.5


def alpha_schedule(n: int, alpha0: float, gamma: float) -> list[float]:
    """最初を α0 から始め、後半で一気に鮮明になるように配る。純ノイズ(α=0)にしない。"""
    return [alpha0 + (1 - alpha0) * (i / (n - 1)) ** gamma for i in range(n)]


def make_noise(size: tuple[int, int], seed: int, grain: int, colored: bool) -> np.ndarray:
    """ノイズ画像。grain は粒の大きさ(1=画素ごと)。見え方が変わるのでE6で比べる対象。"""
    rng = np.random.default_rng(seed)
    w, h = size
    small = (max(1, w // grain), max(1, h // grain))
    shape = (small[1], small[0], 3) if colored else (small[1], small[0], 1)
    noise = rng.integers(0, 256, size=shape, dtype=np.uint8)
    if not colored:
        noise = np.repeat(noise, 3, axis=2)
    if grain > 1:
        noise = np.asarray(Image.fromarray(noise).resize((w, h), Image.NEAREST))
    return noise.astype("float32")


def build_frames(final: Image.Image, alphas: list[float], seed: int,
                 grain: int = 1, colored: bool = True) -> list[Image.Image]:
    base = np.asarray(final.convert("RGB"), dtype="float32")
    noise = make_noise(final.size, seed, grain, colored)
    return [Image.fromarray((base * a + noise * (1 - a)).clip(0, 255).astype("uint8"))
            for a in alphas]


def sheet(frames: list[Image.Image], labels: list[str], cell: int = 256,
          highlight: set[int] | None = None) -> Image.Image:
    highlight = highlight or set()
    img = Image.new("RGB", (cell * len(frames), cell + 20), "black")
    draw = ImageDraw.Draw(img)
    for i, (f, label) in enumerate(zip(frames, labels)):
        img.paste(f.resize((cell, cell)), (cell * i, 20))
        draw.text((cell * i + 4, 5), label, fill="yellow" if i in highlight else "white")
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--finals", nargs="+", type=Path, required=False,
                   help="最終画像。省略時は M1 の結果から拾う")
    p.add_argument("--alpha0", type=float, default=0.15)
    p.add_argument("--gammas", type=float, nargs="+", default=[1.0, 2.0, 3.0],
                   help="難易度のつまみ。並べて比べる")
    p.add_argument("--grain", type=int, default=1, help="ノイズの粒の大きさ")
    p.add_argument("--mono", action="store_true", help="白黒ノイズにする")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = p.parse_args()

    finals = args.finals
    if not finals:
        finals = sorted(Path("experiments/m1_x0_frames/results").glob("*/frames/*/final.png"))
    if not finals:
        raise SystemExit("最終画像が見つからない。先に M1 を実行するか --finals で指定すること。")

    out_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ヒント(5.5秒)と切り替えが重ならないかを確認する。
    gaps = [abs(t - HINT_TIME_S) for t in FRAME_TIMES_S]
    print(f"フレーム時刻: {FRAME_TIMES_S}")
    print(f"ヒント {HINT_TIME_S}秒 と最も近い切り替えとの間隔: {min(gaps):.2f}秒")
    assert min(gaps) >= 0.4, "ヒントとフレーム切り替えが近すぎる(§3.3)"

    n = len(FRAME_TIMES_S)
    for gamma in args.gammas:
        alphas = alpha_schedule(n, args.alpha0, gamma)
        print(f"\nγ={gamma}: α = " + " ".join(f"{a:.2f}" for a in alphas))
        for path in finals:
            name = path.parent.name
            frames = build_frames(Image.open(path), alphas, args.seed, args.grain, not args.mono)
            labels = [f"{t:.0f}s a={a:.2f}" for t, a in zip(FRAME_TIMES_S, alphas)]
            sheet(frames, labels, highlight={5, 6}).save(
                out_dir / f"g{gamma:.1f}_{name}.png")

    print(f"\n書き出し: {out_dir}")
    print("黄色のラベルは、ヒント(5.5秒)を挟む2フレーム。")


if __name__ == "__main__":
    main()
