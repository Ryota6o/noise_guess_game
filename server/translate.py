"""日本語の3語を、画像生成に渡す英文にする。計画書 §3.5。

**英訳を前提にすることは確定している**(E4の実測。日本語直入力は、日本語の文字が
絵に描き込まれる失敗が2倍になる)。

モデルは**すでに事前チェックで読み込んでいる Qwen3-VL を、画像なしで使い回す**。
別の LLM を載せると VRAM が苦しいため(Z-Image 20.4GB + VLM 約18GB で、48GB のうち
残りは約10GB)。

解くべきこと(計画書 §3.5):

- **同音異義語を他の2語の文脈で解く**(橋/箸、紙/神)。
- **3語を自然な英文に組み立てる**。日本語側の助詞の破綻に引きずられない。
- 画像生成に向いた形にする(冠詞つきの名詞句、述語は進行形)。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import torch

from rules import SLOTS

SYSTEM = (
    "You turn three Japanese words into an English prompt for an image generator. "
    "The words fill three slots: subject (who), object (what), predicate (did what). "
    "\n\n"
    "Rules:\n"
    "1. Never translate character by character. 金棒 is a kanabo, a heavy spiked iron club "
    "carried by an oni - it is not a 'gold rod'. 鬼 is a Japanese oni, a horned ogre with "
    "red or blue skin - it is not a 'ghost'.\n"
    "2. When a word names something specific to Japan that an English speaker would not "
    "picture correctly, keep its name and add a short visual description: "
    "'a Japanese oni, a red-skinned horned ogre in a tiger-skin loincloth'.\n"
    "3. Use the other two words as context to resolve homonyms "
    "(橋 bridge vs 箸 chopsticks, 紙 paper vs 神 god).\n"
    "4. For each slot also give a short plain label: a noun phrase with an article for "
    "subject and object ('a dog'), a present participle for the predicate ('chasing').\n"
    "5. The sentence must describe one scene, using the descriptions from rule 2, "
    "and must show the subject doing the action to the object.\n"
    "\n"
    "Reply with JSON only, no explanation, in exactly this form:\n"
    '{"subject": "", "object": "", "predicate": "", "sentence": ""}\n'
    "where subject/object/predicate are the short labels from rule 4, "
    "and sentence is the descriptive prompt from rule 5."
)



@dataclass
class Translation:
    words_en: dict[str, str]
    sentence: str
    ok: bool                       # 3スロットすべて訳せたか
    failed_slots: set[str] = field(default_factory=set)
    raw: str = ""
    seconds: float = 0.0


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class Translator:
    """Qwen3-VL をテキストのみで使う翻訳器。"""

    def __init__(self, vlm_pair, fallback: dict[str, str] | None = None) -> None:
        self.model, self.proc = vlm_pair
        # 辞書引きの控え。LLM が壊れた出力を返したときだけ使う。
        self.fallback = fallback or {}

    def _generate(self, user: str, max_new_tokens: int = 220) -> str:
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": [{"type": "text", "text": user}]}]
        inputs = self.proc.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        return self.proc.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    def translate(self, words_ja: dict[str, str]) -> Translation:
        import time

        user = (f"subject (だれが): {words_ja['subject']}\n"
                f"object (なにを): {words_ja['object']}\n"
                f"predicate (どうした): {words_ja['predicate']}")
        t0 = time.perf_counter()
        raw = self._generate(user)
        seconds = time.perf_counter() - t0

        data = _extract_json(raw) or {}
        words_en: dict[str, str] = {}
        failed: set[str] = set()
        for slot in SLOTS:
            value = str(data.get(slot, "")).strip()
            # 日本語が残っていたら訳せていない。そのまま生成に渡すと、
            # 日本語の文字が絵に描き込まれる(E4で実測済み)。
            if not value or re.search(r"[ぁ-んァ-ヶ一-龥]", value):
                value = self.fallback.get(words_ja[slot], "")
                if not value:
                    failed.add(slot)
                    value = words_ja[slot]
            words_en[slot] = value

        sentence = str(data.get("sentence", "")).strip()
        if not sentence or re.search(r"[ぁ-んァ-ヶ一-龥]", sentence):
            sentence = f"{words_en['subject']} {words_en['predicate']} {words_en['object']}"

        return Translation(words_en=words_en, sentence=sentence, ok=not failed,
                           failed_slots=failed, raw=raw, seconds=round(seconds, 3))
