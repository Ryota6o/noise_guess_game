"""GPU を使わずにサーバーの進行を試すための差し替え。

モデルを読まずに、形だけ同じ `TurnAssets` を返す。
**ルールと進行のテスト専用**で、本番経路では使わない。
"""

from __future__ import annotations

import base64
import io
import random

from PIL import Image

from config import DEFAULT, Config
from gpu import TurnAssets
from rules import SLOTS


class FakeTurnPreparer:
    def __init__(self, config: Config = DEFAULT) -> None:
        self.config = config

    @staticmethod
    def _tiny_jpeg(level: int) -> str:
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (level, level, level)).save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    def prepare_turn(self, words_ja: dict[str, str], seed: int,
                     substitute_pool=None) -> TurnAssets:
        rng = random.Random(seed)
        times = self.config.timing.frame_times_seconds
        choices = {}
        for slot in SLOTS:
            correct = words_ja[slot]
            opts = [{"ja": correct, "en": correct, "role": "correct", "sim": 1.0}]
            opts += [{"ja": f"{correct}_ダミー{i}", "en": f"{correct}_d{i}",
                      "role": r, "sim": 0.9 - 0.1 * i}
                     for i, r in enumerate(("near", "mid", "far"))]
            rng.shuffle(opts)
            choices[slot] = opts
        return TurnAssets(
            words_ja=dict(words_ja), words_en=dict(words_ja),
            prompt_en=" ".join(words_ja[s] for s in SLOTS),
            frames_jpeg_b64=[self._tiny_jpeg(int(255 * i / (len(times) - 1)))
                             for i in range(len(times))],
            frame_times=times, choices=choices,
            timings={"generate": 0.0, "precheck": 0.0, "total": 0.0},
            precheck={s: {"method": "fake", "passed": True} for s in SLOTS})
