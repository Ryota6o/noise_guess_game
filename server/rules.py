"""ゲームのルールエンジン。RULES.md の実装。

**I/O を一切持たない**。通信も時計も画像生成も触らないので、テストで全分岐を確かめられる。
サーバー(app.py)はこのエンジンに出来事を伝え、返ってきた状態を配信するだけ。

正本は `RULES.md`。矛盾を見つけたら、自己判断せず人間に確認すること
(dq-game-rules スキル)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from config import DEFAULT, Config

# RULES.md §1.2: スロットは3つ。補語を追加しない。
SLOTS: tuple[str, str, str] = ("subject", "object", "predicate")
SLOT_LABELS = {"subject": "だれが", "object": "なにを", "predicate": "どうした"}

PlayerId = str


class Phase(str, Enum):
    """RULES.md §2.1 のフェイズ。"""

    INPUT = "input"
    PREPARE = "prepare"
    ANSWER = "answer"
    RESULT = "result"


class TurnOutcome(str, Enum):
    ONGOING = "ongoing"
    SCORED = "scored"          # 誰かが正解して1点入った
    NO_SCORE = "no_score"      # 全員が誤答または時間切れ
    VOID = "void"              # 無効。やり直す(ターン数に数えない)


@dataclass
class Answer:
    """1人の回答。先着の判定はサーバーの受信時刻で行う(RULES.md §4.3、§8)。"""

    player: PlayerId
    picks: dict[str, str]      # slot -> 選んだ語
    received_at: float         # サーバー側の単調増加時刻


@dataclass
class Turn:
    """1ターン。RULES.md §2〜§4。"""

    index: int
    assignments: dict[PlayerId, str]        # player -> 担当スロット
    words: dict[str, str] = field(default_factory=dict)   # slot -> 確定した語
    timed_out: set[PlayerId] = field(default_factory=set)
    # 誤答して解答権を失った人(RULES.md §4.4)
    disqualified: set[PlayerId] = field(default_factory=set)
    answers: list[Answer] = field(default_factory=list)
    winner: PlayerId | None = None
    outcome: TurnOutcome = TurnOutcome.ONGOING
    # 事前チェックで差し替えられたスロットと、その入力者
    substituted: dict[str, PlayerId] = field(default_factory=dict)

    def slot_of(self, player: PlayerId) -> str:
        return self.assignments[player]

    def answerable_slots(self, player: PlayerId) -> list[str]:
        """自分以外の2スロット。自分の語は答えない(RULES.md §4.1)。"""
        own = self.slot_of(player)
        return [s for s in SLOTS if s != own]


@dataclass
class PlayerState:
    player: PlayerId
    round_points: int = 0
    # 語の差し替え回数は**ゲーム単位**で数える(RULES.md §6)。ラウンドでリセットしない。
    substitutions_this_game: int = 0


@dataclass
class Round:
    index: int
    players: list[PlayerId]
    states: dict[PlayerId, PlayerState]
    turns: list[Turn] = field(default_factory=list)
    winner: PlayerId | None = None
    finished: bool = False

    def counted_turns(self) -> int:
        """ターン数の上限に数えるターン(無効ターンは数えない)。"""
        return sum(1 for t in self.turns if t.outcome is not TurnOutcome.VOID)

    def points(self) -> dict[PlayerId, int]:
        return {p: self.states[p].round_points for p in self.players}


class RuleError(Exception):
    """ルール上あり得ない操作。サーバーの実装ミスを早く見つけるために投げる。"""


class Game:
    """1ゲーム。RULES.md §1、§7。"""

    def __init__(self, players: list[PlayerId], rounds: int,
                 config: Config = DEFAULT) -> None:
        s = config.scoring
        if len(players) != s.players_per_group:
            raise RuleError(f"人数は{s.players_per_group}人(渡されたのは{len(players)}人)")
        if not s.rounds_min <= rounds <= s.rounds_max:
            raise RuleError(f"ラウンド数は{s.rounds_min}〜{s.rounds_max}")
        self.config = config
        self.players = list(players)
        self.total_rounds = rounds
        self.round_wins: dict[PlayerId, int] = {p: 0 for p in players}
        self.total_points: dict[PlayerId, int] = {p: 0 for p in players}
        # 語の再利用禁止は**同一ラウンド内のみ**(RULES.md §3.1)。ゲーム全体に広げない。
        self.used_words_this_round: set[str] = set()
        self.substitutions: dict[PlayerId, int] = {p: 0 for p in players}
        self.rounds: list[Round] = []
        self._turn_counter = 0

    # ----- ラウンド -----

    def start_round(self) -> Round:
        if len(self.rounds) >= self.total_rounds:
            raise RuleError("すでに全ラウンドが終わっている")
        states = {p: PlayerState(p, 0, self.substitutions[p]) for p in self.players}
        rnd = Round(index=len(self.rounds), players=list(self.players), states=states)
        self.rounds.append(rnd)
        self.used_words_this_round = set()   # ラウンドが変われば語を再利用できる
        return rnd

    @property
    def current_round(self) -> Round:
        if not self.rounds:
            raise RuleError("ラウンドが始まっていない")
        return self.rounds[-1]

    # ----- ターン -----

    def assign_slots(self, turn_index: int) -> dict[PlayerId, str]:
        """担当スロットはターンごとにローテーションする(RULES.md §1.3)。"""
        return {p: SLOTS[(i + turn_index) % len(SLOTS)]
                for i, p in enumerate(self.players)}

    def start_turn(self) -> Turn:
        rnd = self.current_round
        if rnd.finished:
            raise RuleError("終わったラウンドにターンを足せない")
        turn = Turn(index=self._turn_counter,
                    assignments=self.assign_slots(self._turn_counter))
        self._turn_counter += 1
        rnd.turns.append(turn)
        return turn

    def word_is_reusable(self, word: str) -> bool:
        """同一ラウンド内で使われていない語か(RULES.md §3.1)。"""
        return word not in self.used_words_this_round

    def submit_inputs(self, turn: Turn, words: dict[str, str],
                      timed_out: set[PlayerId] | None = None) -> TurnOutcome:
        """入力フェイズの結果を渡す。

        時間切れの人の語はシステムが補うので(RULES.md §3.3)、`words` は
        3スロットすべて埋まっている前提。
        """
        timed_out = set(timed_out or ())
        unknown = timed_out - set(self.players)
        if unknown:
            raise RuleError(f"知らないプレイヤー: {unknown}")
        if set(words) != set(SLOTS):
            raise RuleError(f"3スロットすべて必要: {sorted(words)}")

        turn.timed_out = timed_out
        turn.words = dict(words)

        # 2人以上が時間切れならターン無効(1人なら成立)
        if len(timed_out) >= self.config.scoring.void_turn_on_timeouts:
            turn.outcome = TurnOutcome.VOID
            return turn.outcome

        self.used_words_this_round |= set(words.values())
        return TurnOutcome.ONGOING

    def is_eligible(self, turn: Turn, player: PlayerId) -> bool:
        """そのターンに回答する資格があるか。

        時間切れの人は回答できない(RULES.md §3.3)。誤答した人は解答権を失う(§4.4)。
        無効になったターンは誰も回答できない。
        """
        if turn.outcome is TurnOutcome.VOID:
            return False
        return player not in turn.timed_out and player not in turn.disqualified

    def can_answer(self, turn: Turn, player: PlayerId) -> bool:
        """今この瞬間、回答を受け付けて**得点になりうる**か。

        先着が決まったあとや、全員が脱落したあとは False。
        ただし、通信中だった回答が直後に届くことはあるので、
        `submit_answer` はそれを例外にせず受け止める。
        """
        return turn.outcome is TurnOutcome.ONGOING and self.is_eligible(turn, player)

    def _picks_are_correct(self, turn: Turn, player: PlayerId,
                           picks: dict[str, str]) -> bool:
        expected = {s: turn.words[s] for s in turn.answerable_slots(player)}
        if set(picks) != set(expected):
            raise RuleError(f"回答すべきスロットが違う: {sorted(picks)} / {sorted(expected)}")

        if picks == expected:
            return True
        # E2 の結果次第で、主語と目的語を入れ替えた回答も正解にする(設定で切り替え)
        if self.config.scoring.accept_swapped_answer and set(picks) == {"subject", "object"}:
            swapped = {"subject": expected["object"], "object": expected["subject"]}
            return picks == swapped
        return False

    def submit_answer(self, turn: Turn, player: PlayerId, picks: dict[str, str],
                      received_at: float) -> bool:
        """回答を受け付ける。正誤を返す。

        得点は**最も早く正解した1人だけ**(RULES.md §4.3)。
        誤答した人はそのターンの解答権を失う(§4.4)。再提出は RuleError。
        """
        if not self.is_eligible(turn, player):
            raise RuleError(f"{player} はこのターン回答する資格がない")
        if turn.outcome is not TurnOutcome.ONGOING:
            # ターンはもう決着している。先着が決まる直前に送られ、わずかに遅れて
            # 届いた回答がこれに当たる。**例外にしない**(通信では普通に起きる)。
            # 得点は入らず、失格にもしない。正誤だけ返す。
            turn.answers.append(Answer(player, dict(picks), received_at))
            return self._picks_are_correct(turn, player, picks)

        turn.answers.append(Answer(player, dict(picks), received_at))
        correct = self._picks_are_correct(turn, player, picks)
        if correct:
            turn.winner = player
            turn.outcome = TurnOutcome.SCORED
            self.current_round.states[player].round_points += 1
        else:
            turn.disqualified.add(player)
            if not self.active_answerers(turn):
                # 全員が誤答または時間切れ。無得点で正解を見せて次へ(RULES.md §4.4)
                turn.outcome = TurnOutcome.NO_SCORE
        return correct

    def active_answerers(self, turn: Turn) -> list[PlayerId]:
        """まだ答えられる人。1人だけ残った場合も、そのターンは続行する(§4.4)。"""
        return [p for p in self.players
                if p not in turn.timed_out and p not in turn.disqualified]

    def finish_turn_by_timeout(self, turn: Turn) -> TurnOutcome:
        """10秒経っても誰も正解しなかった場合。"""
        if turn.outcome is TurnOutcome.ONGOING:
            turn.outcome = TurnOutcome.NO_SCORE
        return turn.outcome

    # ----- 語の差し替えと減点(RULES.md §6) -----

    def record_substitution(self, turn: Turn, slot: str, player: PlayerId) -> bool:
        """事前チェックを通らず語が差し替えられた。減点したかを返す。

        ゲーム中3回目までは何も起きない。4回目以降が減点。
        減点はそのラウンドの得点から1点で、**0未満にはしない**。
        """
        turn.substituted[slot] = player
        self.substitutions[player] += 1
        state = self.current_round.states[player]
        state.substitutions_this_game = self.substitutions[player]

        s = self.config.scoring
        if self.substitutions[player] <= s.free_substitutions_per_game:
            return False
        state.round_points = max(s.min_round_points,
                                 state.round_points - s.substitution_penalty_points)
        return True

    # ----- 決着(RULES.md §7) -----

    def check_round_end(self) -> Round | None:
        """ラウンドが決着したか。決着していれば Round を返す。"""
        rnd = self.current_round
        if rnd.finished:
            return rnd
        s = self.config.scoring
        points = rnd.points()

        reached = [p for p, v in points.items() if v >= s.points_to_win_round]
        if reached:
            # 先取なので、同時に2人が到達することはない
            rnd.winner = reached[0]
        elif rnd.counted_turns() >= s.max_turns_per_round:
            # 10ターンで決着しない場合は最多得点者。同点なら勝者なしの引き分け【現行案】
            best = max(points.values())
            leaders = [p for p, v in points.items() if v == best]
            rnd.winner = leaders[0] if len(leaders) == 1 else None
        else:
            return None

        rnd.finished = True
        if rnd.winner is not None:
            self.round_wins[rnd.winner] += 1
        for p, v in points.items():
            self.total_points[p] += v
        return rnd

    @property
    def finished(self) -> bool:
        return (len(self.rounds) == self.total_rounds
                and self.rounds[-1].finished)

    def result(self) -> dict:
        """最終順位。ラウンド勝利数 → 合計得点 → 同率(RULES.md §7.2)。"""
        if not self.finished:
            raise RuleError("まだゲームが終わっていない")
        best_wins = max(self.round_wins.values())
        leaders = [p for p, v in self.round_wins.items() if v == best_wins]
        if len(leaders) > 1:
            best_points = max(self.total_points[p] for p in leaders)
            leaders = [p for p in leaders if self.total_points[p] == best_points]
        return {"winners": leaders,                 # 複数なら同率
                "round_wins": dict(self.round_wins),
                "total_points": dict(self.total_points)}
