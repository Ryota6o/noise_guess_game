"""1ターン分の出題を用意する。生成・事前チェック・フレーム・選択肢。

計画書 §3.2(線形補間)、§3.4(事前チェック)、RULES.md §5(選択肢)、§8(配信)。

**フレームは全枚数を作り、事前チェックを通してから返す**(dq-realtime-sync スキル)。
生成しながら1枚ずつ流すと、絵に出ない語を弾けなくなる。

重い処理をすべて抱えるので、サーバー本体(app.py)からは `prepare_turn` だけを呼ぶ。
"""

from __future__ import annotations

import base64
import io
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "m3_precheck"))
from dummies import DummyPicker, Word  # noqa: E402

from config import DEFAULT, Config  # noqa: E402
from rules import SLOTS  # noqa: E402

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
VLM_ID = "Qwen/Qwen3-VL-8B-Instruct"
IMAGE_SIZE = 512          # 【確定 2026-10-16】
STEPS = 8
ALPHA0 = 0.15             # 【現行案】最初のフレームの鮮明さ。0にしない
GAMMA = 2.0               # 【現行案】難易度の主要なつまみ。E7 で調整する

CATEGORY_OF_SLOT = {"subject": "subjects", "object": "objects", "predicate": "predicates"}


@dataclass
class TurnAssets:
    """1ターン分の出題一式。サーバーはこれを配るだけ。"""

    words_ja: dict[str, str]
    words_en: dict[str, str]
    prompt_en: str
    frames_jpeg_b64: list[str]
    frame_times: tuple[float, ...]
    choices: dict[str, list[dict]]           # slot -> 4択(表示順はここで固定)
    substituted: dict[str, str] = field(default_factory=dict)  # slot -> 元の語
    timings: dict[str, float] = field(default_factory=dict)
    precheck: dict[str, dict] = field(default_factory=dict)
    failed: bool = False                     # 事前チェックを通せずターン無効


def alpha_schedule(n: int, alpha0: float = ALPHA0, gamma: float = GAMMA) -> list[float]:
    """計画書 §3.2。後半で一気に鮮明になるように配る。純ノイズから始めない。"""
    return [alpha0 + (1 - alpha0) * (i / (n - 1)) ** gamma for i in range(n)]


class TurnPreparer:
    def __init__(self, config: Config = DEFAULT, use_vlm: bool = True,
                 device: str = "cuda") -> None:
        self.config = config
        self.device = device
        self.use_vlm = use_vlm
        self._pipe = None
        self._vlm = None
        self._picker: DummyPicker | None = None
        # 日本語→英語の対応。**本来は LLM で訳す(計画書 §3.5)**。
        # ここは試作用の暫定で、プールにある語だけを引く。
        # プールにない語はそのまま渡すので、日本語が画像に描き込まれる失敗が起きうる。
        self._ja_to_en: dict[str, str] = {}

    # ----- モデルの読み込み(重いので遅延) -----

    @property
    def picker(self) -> DummyPicker:
        if self._picker is None:
            self._picker = DummyPicker(device=self.device)
            for words in self._picker.pool.values():
                for w in words:
                    self._ja_to_en.setdefault(w.ja, w.en)
        return self._picker

    @property
    def pipe(self):
        if self._pipe is None:
            from diffusers import ZImagePipeline
            try:
                self._pipe = ZImagePipeline.from_pretrained(
                    MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=False)
            except TypeError:
                self._pipe = ZImagePipeline.from_pretrained(
                    MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
            self._pipe.to(self.device)
        return self._pipe

    @property
    def vlm(self):
        if self._vlm is None and self.use_vlm:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                VLM_ID, dtype=torch.bfloat16, device_map=self.device).eval()
            self._vlm = (model, AutoProcessor.from_pretrained(VLM_ID))
        return self._vlm

    def warmup(self) -> dict[str, float]:
        """初回の呼び出しは極端に遅い。サーバー起動時に1回走らせて暖めておく。

        実測(2026-10-16): 暖機なしだと1問目の生成が22.3秒、VLMの初回が12.8秒。
        暖機後は生成1.07秒・事前チェック0.11秒で、合計1.19秒。
        **暖機しないと1問目だけ予算(5秒)を20秒以上超える。**
        """
        timings = {}
        t0 = time.perf_counter()
        image = self.generate("a dog chasing a cat", 0)
        timings["generate"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        self._clip_pick(image, ["a dog", "a cat"])
        timings["clip"] = round(time.perf_counter() - t0, 3)
        if self.use_vlm:
            t0 = time.perf_counter()
            self._vlm_pick(image, ["chasing", "eating", "reading", "throwing"])
            timings["vlm"] = round(time.perf_counter() - t0, 3)
        return timings

    # ----- 翻訳(暫定) -----

    def translate(self, slot: str, word_ja: str) -> tuple[str, bool]:
        """日本語の1語を英語にする。(訳, 訳せたか) を返す。

        **LLM に置き換える前の暫定実装**(計画書 §3.5)。プールにある語しか訳せない。

        訳せなかった語を**そのままプロンプトに流してはいけない**。実測(2026-10-16)で
        `a mouse kicking アオマダラウミヘビ` のような混在プロンプトが生成に渡り、
        さらに CLIP が日本語文字列の埋め込みで照合して「通過」と誤判定した。
        訳せない語は事前チェックを通さず、差し替えの対象にする。
        """
        self.picker  # 対応表を用意する
        en = self._ja_to_en.get(word_ja)
        return (en, True) if en else (word_ja, False)

    @staticmethod
    def build_prompt(words_en: dict[str, str]) -> str:
        return f"{words_en['subject']} {words_en['predicate']} {words_en['object']}"

    # ----- 生成 -----

    def generate(self, prompt: str, seed: int) -> Image.Image:
        return self.pipe(prompt=prompt, height=IMAGE_SIZE, width=IMAGE_SIZE,
                         num_inference_steps=STEPS, guidance_scale=0.0,
                         generator=torch.Generator(self.device).manual_seed(seed)
                         ).images[0]

    # ----- 事前チェック(計画書 §3.4) -----

    def _clip_pick(self, image: Image.Image, candidates: list[str]) -> tuple[str, float]:
        p = self.picker
        with torch.no_grad():
            ii = p.proc(images=image, return_tensors="pt").to(self.device)
            out = p.model.get_image_features(**ii)
            f_i = out if torch.is_tensor(out) else out.pooler_output
            f_i = f_i / f_i.norm(dim=-1, keepdim=True)
        sims = (f_i @ p._embed(candidates).T)[0].tolist()
        order = sorted(range(len(candidates)), key=lambda i: -sims[i])
        margin = sims[order[0]] - sims[order[1]] if len(order) > 1 else 1.0
        return candidates[order[0]], margin

    def _vlm_pick(self, image: Image.Image, options: list[str]) -> str | None:
        if not self.use_vlm:
            return None
        model, proc = self.vlm
        letters = "ABCD"[:len(options)]
        listing = "\n".join(f"{c}) {o}" for c, o in zip(letters, options))
        # **自由記述で聞かない**。常識で答えられてしまう(M2 の E2 で実証、E5 で確認済み)
        q = ("Look at the image. What is the main action taking place?\n"
             f"{listing}\nAnswer with a single letter and nothing else.")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": q}]}]
        inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        text = proc.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip().upper()
        m = re.search(rf"\b([{letters}])\b", text)
        return options[letters.index(m.group(1))] if m else None

    def precheck(self, image: Image.Image, words_en: dict[str, str],
                 choices: dict[str, list[dict]], skip_vlm: bool) -> dict[str, dict]:
        """3語が絵に出ているかを、**実際に出す4択で**確かめる(E3の設計)。"""
        result: dict[str, dict] = {}
        for slot in ("subject", "object"):
            options = [o["en"] for o in choices[slot]]
            top, margin = self._clip_pick(image, options)
            threshold = (self.config.precheck.margin_threshold_subject if slot == "subject"
                         else self.config.precheck.margin_threshold_object)
            result[slot] = {"method": "clip", "picked": top, "margin": round(margin, 4),
                            "passed": top == words_en[slot] and margin >= threshold}
        if skip_vlm or not self.use_vlm:
            # 予算を超えそうなときは VLM を省略する経路(計画書 §3.4-6)。
            # CLIP チェックは省略しない。
            result["predicate"] = {"method": "skipped", "passed": True}
        else:
            options = [o["en"] for o in choices["predicate"]]
            picked = self._vlm_pick(image, options)
            result["predicate"] = {"method": "vlm", "picked": picked,
                                   "passed": picked == words_en["predicate"]}
        return result

    # ----- 選択肢(RULES.md §5) -----

    def build_choices(self, words_ja: dict[str, str], words_en: dict[str, str],
                      rng: random.Random) -> dict[str, list[dict]]:
        used_en: set[str] = set()
        choices: dict[str, list[dict]] = {}
        for slot in SLOTS:
            category = CATEGORY_OF_SLOT[slot]
            correct = Word(words_ja[slot], words_en[slot])
            # 主語と目的語に同じ語を並べない
            q = self.picker.pick(category, correct, exclude_en=used_en, rng=rng)
            used_en |= {o["en"] for o in q["options"]}
            choices[slot] = q["options"]
        return choices

    # ----- フレーム(計画書 §3.2: 線形補間) -----

    def build_frames(self, final: Image.Image, seed: int) -> list[str]:
        times = self.config.timing.frame_times_seconds
        alphas = alpha_schedule(len(times))
        rng = np.random.default_rng(seed)
        noise = rng.integers(0, 256, size=(final.height, final.width, 3),
                             dtype=np.uint8).astype("float32")
        base = np.asarray(final.convert("RGB"), dtype="float32")
        out = []
        for a in alphas:
            frame = Image.fromarray((base * a + noise * (1 - a)).clip(0, 255).astype("uint8"))
            buf = io.BytesIO()
            frame.save(buf, format="JPEG", quality=80)
            out.append(base64.b64encode(buf.getvalue()).decode())
        return out

    # ----- 本体 -----

    def prepare_turn(self, words_ja: dict[str, str], seed: int,
                     substitute_pool: dict[str, list[str]] | None = None) -> TurnAssets:
        """1ターン分を用意する。事前チェックを通らなければ語を差し替えて再生成する。"""
        rng = random.Random(seed)
        words_ja = dict(words_ja)
        substituted: dict[str, str] = {}
        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        for attempt in range(self.config.precheck.retry_limit + 1):
            translated = {s: self.translate(s, w) for s, w in words_ja.items()}
            words_en = {s: en for s, (en, _) in translated.items()}
            untranslated = {s for s, (_, ok) in translated.items() if not ok}
            prompt = self.build_prompt(words_en)

            t0 = time.perf_counter()
            image = self.generate(prompt, seed + attempt)
            timings["generate"] = round(time.perf_counter() - t0, 3)

            choices = self.build_choices(words_ja, words_en, rng)

            elapsed = time.perf_counter() - t_start
            skip_vlm = (self.config.precheck.skip_vlm_when_over_budget
                        and elapsed > self.config.timing.prepare_budget_seconds * 0.6)
            t0 = time.perf_counter()
            checks = self.precheck(image, words_en, choices, skip_vlm)
            # 訳せなかった語は、CLIPの判定を信用せず必ず落とす。
            # 日本語文字列どうしの照合は意味を持たないため。
            for slot in untranslated:
                checks[slot] = {"method": "untranslated", "passed": False,
                                "picked": None}
            timings["precheck"] = round(time.perf_counter() - t0, 3)

            failed_slots = [s for s, r in checks.items() if not r["passed"]]
            if not failed_slots or attempt == self.config.precheck.retry_limit:
                frames = self.build_frames(image, seed)
                timings["total"] = round(time.perf_counter() - t_start, 3)
                return TurnAssets(
                    words_ja=words_ja, words_en=words_en, prompt_en=prompt,
                    frames_jpeg_b64=frames,
                    frame_times=self.config.timing.frame_times_seconds,
                    choices=choices, substituted=substituted, timings=timings,
                    precheck=checks, failed=bool(failed_slots))

            # 通らなかった語を差し替える。差し替え先は、過去に通った語のプールから同カテゴリ
            for slot in failed_slots:
                pool = (substitute_pool or {}).get(CATEGORY_OF_SLOT[slot])
                if not pool:
                    pool = [w.ja for w in self.picker.pool[CATEGORY_OF_SLOT[slot]]]
                candidate = rng.choice([w for w in pool if w != words_ja[slot]])
                substituted[slot] = words_ja[slot]
                words_ja[slot] = candidate

        raise RuntimeError("到達しない")
