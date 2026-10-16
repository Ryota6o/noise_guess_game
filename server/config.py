"""ゲームの設定値。

RULES.md の【現行案】【未決定】の項目を、**コードに埋め込まずここに集める**
(dq-game-rules スキルの指示)。変更が1行で済むようにするため。

【確定】の項目も、意味が分かるように名前を付けてここに置くが、
**変更には人間の承認が要る**。コメントにラベルを書いてある。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Timing:
    """RULES.md §2.1 フェイズ、§2.2 回答フェイズ内のタイムライン。"""

    input_seconds: float = 30.0              # 【確定 2026-10-16】全員が入力したら即終了
    #   当初は10秒。「何を出題するか」を考えるのに足りないため30秒にした。
    prepare_budget_seconds: float = 5.0      # 【現行案】生成 + 事前チェック
    answer_seconds: float = 10.0             # 【確定】
    result_seconds: float = 5.0              # 【現行案】
    hint_at_seconds: float = 5.5             # 【確定(タイミング)】
    last_frame_by_seconds: float = 7.0       # 【現行案】見てタップする時間を残す

    # §3.2(計画書): 0〜7秒を1秒刻みで8フレーム。
    # 切り替えが5.0秒と6.0秒になり、5.5秒のヒントが両隣から最も離れる。
    frame_times_seconds: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)

    def hint_gap_seconds(self) -> float:
        """ヒントと、最も近いフレーム切り替えとの間隔。"""
        return min(abs(t - self.hint_at_seconds) for t in self.frame_times_seconds)


@dataclass(frozen=True)
class Scoring:
    """RULES.md §1.2、§4、§6、§7。"""

    players_per_group: int = 3               # 【確定】
    points_to_win_round: int = 3             # 【確定】3ポイント先取
    max_turns_per_round: int = 10            # 【確定】最長7ターン + 無得点3ターン
    rounds_min: int = 1                      # 【確定】
    rounds_max: int = 5                      # 【確定】

    # 【確定】2スロットを両方選んで送信し、両方正解で1点(§4.2 は【現行案】扱い)
    require_both_slots_correct: bool = True

    # 【要確認】E2(主語と目的語の役割の再現)の結果次第。
    # 2026-10-16: 企画者の判断で「役割は描き分けられている」= 対策不要とした。
    # 絵から役割が読めないと分かった場合は True にする(計画書 §8-14、RULES.md §11)。
    accept_swapped_answer: bool = False

    # 【確定】語の差し替えはゲーム中3回目まで減点なし。4回目以降が減点対象。
    free_substitutions_per_game: int = 3
    # 【現行案】減点はそのラウンドの得点から1点。0未満にはしない。
    substitution_penalty_points: int = 1
    min_round_points: int = 0

    # 【現行案】入力の時間切れが2人以上ならターン無効。無効なターンはターン数に数えない。
    void_turn_on_timeouts: int = 2
    count_void_turns: bool = False


@dataclass(frozen=True)
class Choices:
    """RULES.md §5 選択肢(ダミー)の作り方。"""

    options_per_slot: int = 4                # 【現行案】正解1 + 近い1 + 中くらい1 + 遠い1
    # 【現行案】難易度の主要なつまみ。experiments/m3_precheck/dummies.py と揃える。
    # 【要検証】CLIP空間の近さは、絵としての紛らわしさと一致しない。
    # 「猫」の近傍にロボットや車がネズミより上位で入る(2026-10-16)。E7 で見直す。
    near_band: tuple[float, float] = (0.00, 0.12)
    mid_band: tuple[float, float] = (0.40, 0.60)
    far_band: tuple[float, float] = (0.85, 1.00)
    synonym_ceiling: float = 0.95            # これ以上は正解になりうるとみなして除外
    # 【要検証】述語は類似度が0.83〜0.90に圧縮され、類義語が混入する(M3のE5で実証)。
    # 述語だけ上限を下げる必要があるが、値は未決定。
    synonym_ceiling_predicate: float = 0.95


@dataclass(frozen=True)
class Precheck:
    """RULES.md §6、計画書 §3.4。閾値は experiments/m3_precheck/FINDINGS.md 参照。"""

    # 【現行案】4択で正解が1位なら通す。マージン閾値は0から始める。
    # 上げると差し替えが増え、4回目から減点が頻発する(E3のトレードオフ表)。
    margin_threshold_subject: float = 0.0
    margin_threshold_object: float = 0.0
    retry_limit: int = 1                     # 【現行案】やり直しは1回まで

    # 差し替えの対象にするスロット【現行案 2026-10-16】。
    # 3スロット全部を条件にすると、個々が89%・100%・89%でも**全部通るのは79%**になり、
    # 5問に1問で差し替えが起きる。しかも落ちた件の多くは誤判定で、
    # 「男性が犬を洗う」は男性が写っているのにCLIPが farmer を選んで落ちた。
    #
    # **述語を外すと21%→11%に半減する。** 動詞の判定は特に誤りやすく、
    # 述語の担当者だけが不利になる(計画書 §7 のリスク表)。
    # 述語のVLM判定は記録だけ残し、差し替えには使わない。
    substitute_slots: tuple[str, ...] = ("subject", "object")

    # 事前チェックそのものを止める逃げ道。**既定では止めない。**
    # 止めると「絵に出ない語を入れた人が有利」なゲームになる(計画書 §3.4)。
    enabled: bool = True
    skip_vlm_when_over_budget: bool = True   # 【現行案】CLIPチェックは省略しない


@dataclass(frozen=True)
class Config:
    timing: Timing = field(default_factory=Timing)
    scoring: Scoring = field(default_factory=Scoring)
    choices: Choices = field(default_factory=Choices)
    precheck: Precheck = field(default_factory=Precheck)


DEFAULT = Config()
