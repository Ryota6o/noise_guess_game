#!/usr/bin/env python3
"""E1(前半): Z-Image-Turbo の生成時間を解像度別に実測する。

計画書 §6 の E1、マイルストーン M0 に対応する。知りたいのは次の4点。

1. 解像度別の end-to-end 生成時間。1ターンの予算は「生成 + 事前チェックで5秒以内」なので、
   生成だけで何秒使えるかを決める材料にする(RULES.md §2.1)。
2. 工程ごとの内訳(テキストエンコード / ノイズ除去 / VAEデコード)。
   テキストエンコーダは Qwen3(4B)で、ここが重いと解像度を下げても効かない。
3. **実際の NFE(transformer の forward 回数)**。計画書は「公式サンプルは
   num_inference_steps=9 だが評価は8回」と書いているが、diffusers のドキュメントの例は
   num_inference_steps=8 を使っており、記述が食い違っている。取れる x0 予測の枚数は
   フレーム設計(M1)の上限そのものなので、推測せず forward 回数を数えて確定させる。
4. ピークVRAM。CLIP・VLM・LLM を同居させる余地があるかの判断材料(計画書 §4.2)。

出力は results/<timestamp>/ に JSON と Markdown、sanity check 用のサンプル画像。
計測条件(GPU、ドライバ、ライブラリ版、同居プロセス)も一緒に残す。条件のない数値は
役に立たないため(CLAUDE.md「数値と事実の扱い」)。

使い方:
    .venv/bin/python experiments/e1_generation_speed/measure_speed.py
    .venv/bin/python experiments/e1_generation_speed/measure_speed.py --resolutions 512 768 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

# ゲームの文は「だれが・なにを・どうした」の3語。プロンプト長はテキストエンコードの時間に
# 効くので、実際にゲームで流れるのと同じ長さの英文で測る(計画書 §3.5: 英訳を前提)。
DEFAULT_PROMPT = "A dog chases a cat"


@dataclass
class StageTimer:
    """1工程の呼び出し回数と累積時間。GPUを同期して測るので、わずかに遅くなる。"""

    name: str
    calls: int = 0
    total_s: float = 0.0
    per_call_s: list[float] = field(default_factory=list)
    _start: float | None = None

    def begin(self) -> None:
        torch.cuda.synchronize()
        self._start = time.perf_counter()

    def end(self) -> None:
        torch.cuda.synchronize()
        if self._start is None:
            return
        dt = time.perf_counter() - self._start
        self.total_s += dt
        self.per_call_s.append(dt)
        self.calls += 1
        self._start = None

    def reset(self) -> None:
        self.calls = 0
        self.total_s = 0.0
        self.per_call_s = []
        self._start = None


class CallCounter:
    """forward の回数だけを数える。時間は測らないので同期もせず、計測を乱さない。"""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_args, **_kwargs) -> None:
        self.calls += 1

    def reset(self) -> None:
        self.calls = 0


def gpu_environment() -> dict:
    """GPU名・ドライバ・同居プロセスを記録する。同じGPUで他の処理が動いていると遅くなる。"""
    info: dict = {}
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        info["total_vram_mib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**20)
    try:
        info["driver_version"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        # 計測開始時点でGPUを使っていた他プロセス。空なら専有できている。
        info["other_gpu_processes_at_start"] = procs.splitlines() if procs else []
        info["used_vram_mib_at_start"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info["driver_version"] = "unknown"
    return info


def library_versions() -> dict:
    import diffusers
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
    }


def load_pipeline(model_id: str, dtype: torch.dtype):
    """Z-Image-Turbo を読み込む。diffusers は dtype 引数の名前を移行中なので両方試す。"""
    from diffusers import ZImagePipeline

    t0 = time.perf_counter()
    try:
        pipe = ZImagePipeline.from_pretrained(model_id, dtype=dtype, low_cpu_mem_usage=False)
    except TypeError:
        pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=False)
    pipe.to("cuda")
    torch.cuda.synchronize()
    return pipe, time.perf_counter() - t0


def measure(pipe, prompt: str, height: int, width: int, steps: int, repeats: int,
            warmup: int, seed: int, sample_path: Path | None,
            breakdown: bool) -> dict:
    """1つの設定について、warmup のあと repeats 回測る。

    breakdown=True のときは工程ごとに GPU 同期を挟むため、end-to-end の数値はわずかに
    悪化する。予算の判断には breakdown=False の数値を使う。
    """
    nfe = CallCounter()
    handles = [pipe.transformer.register_forward_hook(nfe)]

    timers: dict[str, StageTimer] = {}
    if breakdown:
        timers = {k: StageTimer(k) for k in ("text_encode", "denoise_step", "vae_decode")}
        handles += [
            pipe.text_encoder.register_forward_pre_hook(lambda *_: timers["text_encode"].begin()),
            pipe.text_encoder.register_forward_hook(lambda *_: timers["text_encode"].end()),
            pipe.transformer.register_forward_pre_hook(lambda *_: timers["denoise_step"].begin()),
            pipe.transformer.register_forward_hook(lambda *_: timers["denoise_step"].end()),
        ]
        original_decode = pipe.vae.decode

        def timed_decode(*args, **kwargs):
            timers["vae_decode"].begin()
            out = original_decode(*args, **kwargs)
            timers["vae_decode"].end()
            return out

        pipe.vae.decode = timed_decode

    def run(seed_offset: int):
        return pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,  # Turbo は CFG 不要(計画書 §3.1)
            generator=torch.Generator("cuda").manual_seed(seed + seed_offset),
        )

    try:
        for i in range(warmup):
            run(1000 + i)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        nfe.reset()
        for t in timers.values():
            t.reset()

        wall_times: list[float] = []
        result = None
        for i in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = run(i)
            torch.cuda.synchronize()
            wall_times.append(time.perf_counter() - t0)

        peak_mib = round(torch.cuda.max_memory_allocated() / 2**20)
        if sample_path is not None and result is not None:
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(sample_path)
    finally:
        for h in handles:
            h.remove()
        if breakdown:
            pipe.vae.decode = original_decode

    wall_times.sort()
    out = {
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "repeats": repeats,
        "warmup": warmup,
        "breakdown_instrumented": breakdown,
        "wall_s": {
            "median": round(wall_times[len(wall_times) // 2], 4),
            "mean": round(sum(wall_times) / len(wall_times), 4),
            "min": round(wall_times[0], 4),
            "max": round(wall_times[-1], 4),
            "all": [round(t, 4) for t in wall_times],
        },
        # NFE = transformer の forward 回数 / 試行回数。これが取れる x0 予測の上限枚数。
        "nfe_per_image": nfe.calls / repeats,
        "peak_vram_mib": peak_mib,
    }
    if breakdown:
        out["stage_s_per_image"] = {
            name: {
                "total": round(t.total_s / repeats, 4),
                "calls": t.calls / repeats,
                "per_call_mean": round(t.total_s / t.calls, 4) if t.calls else None,
            }
            for name, t in timers.items()
        }
    return out


def write_summary(path: Path, payload: dict) -> None:
    """人が読む用の Markdown。数値は報告値ではなく、この環境の実測値であることを明記する。"""
    env = payload["environment"]
    lines = [
        "# E1(前半): Z-Image-Turbo 生成時間の実測",
        "",
        f"- 計測日時: {payload['measured_at']}",
        f"- GPU: {env['gpu'].get('device_name', '?')} "
        f"(VRAM {env['gpu'].get('total_vram_mib', '?')} MiB, "
        f"driver {env['gpu'].get('driver_version', '?')})",
        f"- torch {env['libraries']['torch']} / cuda {env['libraries']['torch_cuda']} / "
        f"diffusers {env['libraries']['diffusers']} / transformers {env['libraries']['transformers']}",
        f"- モデル: `{payload['model_id']}` dtype={payload['dtype']}",
        f"- プロンプト: `{payload['prompt']}`",
        f"- 計測開始時にGPUを使っていた他プロセス: "
        f"{env['gpu'].get('other_gpu_processes_at_start') or 'なし'}",
        f"- パイプライン読み込み時間: {payload['pipeline_load_s']} 秒(初回はダウンロードを含む)",
        "",
        "**これはこの環境での実測値であり、報告値ではない。**",
        "",
        "## end-to-end 生成時間(予算判断に使う数値)",
        "",
        "| 解像度 | steps | NFE | 中央値 | 平均 | 最小 | 最大 | ピークVRAM |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["runs"]:
        if r["breakdown_instrumented"]:
            continue
        w = r["wall_s"]
        lines.append(
            f"| {r['width']}x{r['height']} | {r['num_inference_steps']} | {r['nfe_per_image']:g} | "
            f"**{w['median']:.2f}s** | {w['mean']:.2f}s | {w['min']:.2f}s | {w['max']:.2f}s | "
            f"{r['peak_vram_mib']} MiB |"
        )

    breakdowns = [r for r in payload["runs"] if r["breakdown_instrumented"]]
    if breakdowns:
        lines += [
            "",
            "## 工程ごとの内訳(参考)",
            "",
            "工程の境界でGPUを同期して測るため、end-to-end の数値は上の表より悪化する。",
            "内訳の比率を見るためのもの。",
            "",
            "| 解像度 | steps | 合計 | テキストエンコード | ノイズ除去(計) | 1ステップ | VAEデコード |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in breakdowns:
            s = r["stage_s_per_image"]
            lines.append(
                f"| {r['width']}x{r['height']} | {r['num_inference_steps']} | "
                f"{r['wall_s']['median']:.2f}s | {s['text_encode']['total']:.2f}s | "
                f"{s['denoise_step']['total']:.2f}s | "
                f"{s['denoise_step']['per_call_mean']:.3f}s | {s['vae_decode']['total']:.2f}s |"
            )

    lines += [
        "",
        "## 判断の材料",
        "",
        "- 1ターンの予算は「生成 + 事前チェックで5秒以内」(RULES.md §2.1、計画書 §4.4)。",
        "  生成が何秒で収まるかで、事前チェック(CLIP・VLM)に回せる時間が決まる。",
        "- NFE の列が、M1 で取れる x0 予測の**上限枚数**。見せたいフレームは6〜7枚が目安。",
        "- ピークVRAM と 48GB の差が、CLIP・VLM・ダミー生成用LLMを同居させられる余地。",
        "",
        "生データは同じディレクトリの `results.json`、サンプル画像は `samples/`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--resolutions", type=int, nargs="+", default=[512, 640, 768, 1024],
                   help="正方形の一辺。計画書の現行案は512〜768、1024は報告値との比較用")
    p.add_argument("--steps", type=int, nargs="+", default=[8, 9],
                   help="num_inference_steps。8と9で実NFEが変わるかを確認する")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2, help="初回はCUDAカーネルのコンパイル等で遅い")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-breakdown", action="store_true", help="工程別の内訳を測らない")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).parent / "results")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA が使えない。GPUの状態を確認すること。")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {"gpu": gpu_environment(), "libraries": library_versions(),
           "host": platform.platform()}
    print(f"GPU: {env['gpu'].get('device_name')} / diffusers {env['libraries']['diffusers']}")
    print(f"他プロセス: {env['gpu'].get('other_gpu_processes_at_start') or 'なし'}")

    print(f"読み込み中: {args.model}(初回はモデルのダウンロードで時間がかかる)")
    pipe, load_s = load_pipeline(args.model, torch.bfloat16)
    print(f"読み込み完了: {load_s:.1f} 秒")

    runs = []
    # まず steps の違いで実NFEが変わるかを、最小の解像度で確認する。
    for steps in args.steps:
        for res in args.resolutions:
            print(f"[end-to-end] {res}x{res} steps={steps} ...", flush=True)
            r = measure(
                pipe, args.prompt, res, res, steps, args.repeats, args.warmup, args.seed,
                sample_path=out_dir / "samples" / f"{res}px_steps{steps}.png",
                breakdown=False,
            )
            print(f"  中央値 {r['wall_s']['median']:.2f}s / NFE {r['nfe_per_image']:g} / "
                  f"peak {r['peak_vram_mib']} MiB")
            runs.append(r)

    if not args.no_breakdown:
        for res in args.resolutions:
            steps = args.steps[0]
            print(f"[内訳] {res}x{res} steps={steps} ...", flush=True)
            r = measure(
                pipe, args.prompt, res, res, steps, args.repeats, args.warmup, args.seed,
                sample_path=None, breakdown=True,
            )
            s = r["stage_s_per_image"]
            print(f"  text {s['text_encode']['total']:.2f}s / "
                  f"denoise {s['denoise_step']['total']:.2f}s / "
                  f"vae {s['vae_decode']['total']:.2f}s")
            runs.append(r)

    payload = {
        "experiment": "E1-generation-speed",
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": args.model,
        "dtype": "bfloat16",
        "prompt": args.prompt,
        "pipeline_load_s": round(load_s, 1),
        "environment": env,
        "runs": runs,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(out_dir / "summary.md", payload)
    print(f"\n書き出し: {out_dir}/summary.md")


if __name__ == "__main__":
    main()
