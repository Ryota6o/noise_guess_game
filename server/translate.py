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
    "You turn three Japanese words into an English prompt for an image generator, "
    "and into one natural Japanese sentence.\n"
    "The words fill three slots: subject (who), object (what), predicate (did what). "
    "The picture has to show all three clearly, because players guess the words from it.\n\n"
    "What players actually type:\n"
    "- The object slot is NOT always a を-object. It may already carry its own particle "
    "and play another role: 針で is an instrument, 公園で is a place, "
    "腸をだしながら is a manner. Read the particle and use that role. "
    "Never glue を onto something that already ends in a particle.\n"
    "- The predicate slot may be a noun: 切腹 means committing seppuku. "
    "Treat whatever is in that slot as the action.\n"
    "- Build the Japanese sentence so it reads naturally with these parts as given. "
    "Do not invent particles that break it.\n\n"
    "Rules for the English prompt:\n"
    "1. Never translate character by character. 金棒 is a kanabo, a heavy spiked iron club "
    "carried by an oni - it is not a 'gold rod'. 鬼 is a Japanese oni, a horned ogre with "
    "red or blue skin - it is not a 'ghost'.\n"
    "2. When a word names something specific to Japan that an English speaker would not "
    "picture correctly, keep its name and add a short visual description: "
    "'a Japanese oni, a red-skinned horned ogre in a tiger-skin loincloth'.\n"
    "3. Describe the action as a visible pose. Say what the body is doing: which limbs, "
    "what posture, which way things move. Not 'pushing a clock' but 'leaning forward with "
    "both palms flat against the clock, arms straight, pushing it'. A named action on its "
    "own tends to come out as someone standing still.\n"
    "4. Make sure the object is visibly present and separate. If it would otherwise be "
    "hidden or merge into something else, say where it is and what it looks like.\n"
    "5. If the object is the instrument of the action, it must be the ONLY instrument. "
    "Say so explicitly: 'using a banana as the blade, no knife, no sword anywhere'. "
    "Otherwise the model adds the tool the action normally needs.\n"
    "6. Keep the number. 子供たち means several children, not one.\n"
    "7. Keep absurd combinations literal. If the words make a strange scene, draw that "
    "strange scene; do not soften it into something ordinary. A cartoon look is fine for "
    "things that cannot be photographed.\n"
    "8. Use the other two words as context to resolve homonyms "
    "(橋 bridge vs 箸 chopsticks, 紙 paper vs 神 god).\n"
    "8b. The scene contains exactly the people and things named by the three words. "
    "Never introduce anyone else. Never write 'a child', 'a kid' or 'a baby' unless one "
    "of the three words says so. Any body part mentioned belongs to the subject.\n"
    "8c. Do not add gore, wounds, blood or sexual content that the words do not already "
    "state. This is a party game: keep the scene silly rather than graphic. If the words "
    "are violent, show the pose and leave the injury out.\n"
    "8d. Keep the English sentence under 40 words. Long prompts cost time and do not "
    "improve the picture.\n"
    "9. For each slot also give a short plain label in English: a noun phrase with an "
    "article for subject and object ('a needle' for 針で), a present participle for the "
    "predicate ('committing seppuku' for 切腹). Keep these short - they are used to check "
    "the picture, not to draw it.\n"
    "\n"
    "Reply with JSON only, no explanation, in exactly this form:\n"
    '{"subject": "", "object": "", "predicate": "", "sentence": "", "sentence_ja": ""}\n'
    "where subject/object/predicate are the short English labels from rule 9, "
    "sentence is the descriptive English prompt, and sentence_ja is the natural "
    "Japanese sentence made from the three words exactly as the players typed them."
)



@dataclass
class Translation:
    words_en: dict[str, str]
    sentence: str
    # プレイヤーに見せる日本語の文。助詞を機械的に貼ると
    # 「バナナでを切腹する」のように壊れるので、LLM に組み立てさせる。
    sentence_ja: str
    ok: bool                       # 3スロットすべて訳せたか
    failed_slots: set[str] = field(default_factory=set)
    raw: str = ""
    seconds: float = 0.0


# 入力に無いのに英文へ紛れ込みやすい語。実測(2026-10-18)で、
# 「侍 / 腸をだしながら / 笑う」に対して LLM が繰り返し "a child's intestines" を出した。
# 指示で禁じても止まらなかったため、コード側で検査する。
UNINVITED_PEOPLE = ("child", "kid", "baby", "infant", "toddler", "girl", "boy")
# 上の語が入力側にあれば、英文に出てきても問題ない
JA_PEOPLE_HINTS = ("子", "赤ちゃん", "少年", "少女", "幼", "ベビー", "キッズ")


def has_uninvited_person(sentence: str, words_ja: dict[str, str]) -> str | None:
    """入力に無い人物(特に子供)が英文に紛れていないか。混入していればその語を返す。"""
    joined_ja = "".join(words_ja.values())
    if any(h in joined_ja for h in JA_PEOPLE_HINTS):
        return None
    lowered = sentence.lower()
    for term in UNINVITED_PEOPLE:
        if re.search(rf"\b{term}'?s?\b", lowered):
            return term
    return None


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

    def _generate(self, user: str, max_new_tokens: int = 200) -> str:
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
        intruder = has_uninvited_person(raw, words_ja)
        if intruder:
            # 1度だけ、名指しで直させる。それでも消えなければ下で落とす。
            raw = self._generate(
                user + f"\n\nYour previous answer mentioned a {intruder}, which is not "
                "one of the three words. Rewrite without it. Any body part belongs to "
                "the subject.")
        seconds = time.perf_counter() - t0

        data = _extract_json(raw) or {}
        words_en: dict[str, str] = {}
        failed: set[str] = set()
        for slot in SLOTS:
            value = str(data.get(slot, "")).strip()
            # 短いラベルは事前チェックの4択に使う。LLM が説明まで書いてくることがあるので
            # (「鬼」で a Japanese oni, a red-skinned horned ogre... と返した)、
            # 最初の読点までに切る。長い説明とプールの短い語を並べると照合が歪む。
            value = value.split(",")[0].strip()
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

        # 作り直しても消えなければ、説明を捨てて素の文に落とす。
        # 絵は地味になるが、入力に無い人物を描くよりはよい。
        if has_uninvited_person(sentence, words_ja):
            sentence = (f"{words_en['subject']} {words_en['predicate']} "
                        f"{words_en['object']}")

        sentence_ja = str(data.get("sentence_ja", "")).strip()
        if not sentence_ja:
            # 最後の手段。助詞がすでに付いている語には を を足さない。
            obj = words_ja["object"]
            particle = "" if obj and obj[-1] in "でにへとやからまでをもはがより" else "を"
            sentence_ja = f"{words_ja['subject']}が{obj}{particle}{words_ja['predicate']}"

        return Translation(words_en=words_en, sentence=sentence, sentence_ja=sentence_ja,
                           ok=not failed, failed_slots=failed, raw=raw,
                           seconds=round(seconds, 3))
