"""ルールエンジンのテスト。

**RULES.md §9「実装時に外してはいけない原則」の12項目を、1項目ずつ検査する。**
各テストの名前と docstring に、対応する項目の番号を書いてある。
ルールを変えるときは RULES.md を先に直し、ここも一緒に直すこと。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "server"))

from config import Config, Scoring, Timing, DEFAULT          # noqa: E402
from rules import SLOTS, Game, Phase, RuleError, TurnOutcome  # noqa: E402

P = ["alice", "bob", "carol"]


def new_game(rounds: int = 1, **scoring_overrides) -> Game:
    cfg = Config(scoring=Scoring(**scoring_overrides)) if scoring_overrides else DEFAULT
    return Game(P, rounds, cfg)


def play_inputs(game: Game, words=None, timed_out=None):
    turn = game.start_turn()
    words = words or {"subject": f"s{turn.index}", "object": f"o{turn.index}",
                      "predicate": f"v{turn.index}"}
    game.submit_inputs(turn, words, timed_out)
    return turn


def correct_picks(game: Game, turn, player):
    return {s: turn.words[s] for s in turn.answerable_slots(player)}


# ---- 原則1: 得点は「最も早く正解した1人」だけ ----

def test_only_first_correct_answer_scores():
    """§9-1。2位以下は0点。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g)
    assert g.submit_answer(t, "alice", correct_picks(g, t, "alice"), received_at=1.0)
    assert g.submit_answer(t, "bob", correct_picks(g, t, "bob"), received_at=1.5)
    pts = g.current_round.points()
    assert pts["alice"] == 1
    assert pts["bob"] == 0 and pts["carol"] == 0
    assert t.winner == "alice"


def test_no_points_for_the_player_who_supplied_a_word():
    """§9-1。出題者加点は廃止済み。語を出しただけでは1点も入らない。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g)
    g.finish_turn_by_timeout(t)
    assert set(g.current_round.points().values()) == {0}


# ---- 原則2: 誤答したら解答権を失う ----

def test_wrong_answer_loses_the_right_to_answer():
    """§9-2。再提出は RuleError。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g)
    wrong = dict(correct_picks(g, t, "alice"))
    wrong[list(wrong)[0]] = "まったく別の語"
    assert g.submit_answer(t, "alice", wrong, received_at=1.0) is False
    assert g.can_answer(t, "alice") is False
    with pytest.raises(RuleError):
        g.submit_answer(t, "alice", correct_picks(g, t, "alice"), received_at=2.0)


def test_last_remaining_player_continues_alone():
    """§9-7(RULES.md §4.4)。2人が誤答で失格しても、残った1人でターンを続ける。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g)
    for p in ("alice", "bob"):
        wrong = dict(correct_picks(g, t, p))
        wrong[list(wrong)[0]] = "別の語"
        g.submit_answer(t, p, wrong, received_at=1.0)
    assert g.active_answerers(t) == ["carol"]
    assert t.outcome is TurnOutcome.ONGOING          # 打ち切らない
    assert g.submit_answer(t, "carol", correct_picks(g, t, "carol"), received_at=9.9)
    assert g.current_round.points()["carol"] == 1


def test_all_wrong_ends_the_turn_without_points():
    """RULES.md §4.4。全員が誤答なら無得点で次へ。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g)
    for p in P:
        wrong = dict(correct_picks(g, t, p))
        wrong[list(wrong)[0]] = "別の語"
        g.submit_answer(t, p, wrong, received_at=1.0)
    assert t.outcome is TurnOutcome.NO_SCORE
    assert set(g.current_round.points().values()) == {0}


# ---- 原則3: 自分のスロットは答えない ----

def test_a_player_never_answers_their_own_slot():
    """§9-3。回答は常に自分以外の2スロット。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g)
    for p in P:
        slots = t.answerable_slots(p)
        assert len(slots) == 2
        assert t.slot_of(p) not in slots


def test_answering_the_wrong_slot_set_is_rejected():
    """§9-3。自分のスロットを含む回答は RuleError。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g)
    bad = {s: t.words[s] for s in SLOTS}          # 3スロット全部
    with pytest.raises(RuleError):
        g.submit_answer(t, "alice", bad, received_at=1.0)


# ---- 原則4: 語の再利用禁止は同一ラウンド内のみ ----

def test_word_reuse_is_blocked_within_a_round_only():
    """§9-4。ゲーム全体に広げない(動詞の語彙が枯渇するため)。"""
    g = new_game(rounds=2)
    g.start_round()
    play_inputs(g, {"subject": "犬", "object": "猫", "predicate": "追いかける"})
    assert g.word_is_reusable("犬") is False
    g.current_round.finished = True
    g.start_round()
    assert g.word_is_reusable("犬") is True       # ラウンドが変われば使える


# ---- 原則8: 入力の時間切れは、1人なら成立、2人以上なら無効 ----

def test_one_timeout_keeps_the_turn_alive():
    """§9-8。時間切れの人だけが回答できない。減点もしない。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g, timed_out={"carol"})
    assert t.outcome is TurnOutcome.ONGOING
    assert g.can_answer(t, "carol") is False
    assert g.can_answer(t, "alice") is True
    assert g.current_round.points()["carol"] == 0


def test_two_timeouts_void_the_turn():
    """§9-8。2人以上で無効。無効ターンはターン数に数えない(§9の最大10ターン)。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g, timed_out={"bob", "carol"})
    assert t.outcome is TurnOutcome.VOID
    assert g.current_round.counted_turns() == 0


def test_void_turn_does_not_consume_the_word():
    """無効ターンの語は、同一ラウンドで使い直せる(ターンごとやり直すため)。"""
    g = new_game()
    g.start_round()
    play_inputs(g, {"subject": "犬", "object": "猫", "predicate": "追う"},
                timed_out={"bob", "carol"})
    assert g.word_is_reusable("犬") is True


# ---- 原則7: 語の差し替えは3回目まで減点なし ----

def test_first_three_substitutions_are_free_then_deduct():
    """§9-7。4回目以降のみ減点。回数はゲーム単位で数える。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g)
    g.submit_answer(t, "alice", correct_picks(g, t, "alice"), received_at=1.0)
    assert g.current_round.points()["alice"] == 1

    for _ in range(3):
        assert g.record_substitution(t, "subject", "alice") is False
    assert g.current_round.points()["alice"] == 1       # ここまで無傷
    assert g.record_substitution(t, "subject", "alice") is True
    assert g.current_round.points()["alice"] == 0


def test_penalty_never_goes_below_zero():
    """§9-7。0未満にはしない。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g)
    for _ in range(8):
        g.record_substitution(t, "subject", "bob")
    assert g.current_round.points()["bob"] == 0


def test_substitution_count_survives_a_new_round():
    """§9-7。ラウンドをまたいでもリセットしない。"""
    g = new_game(rounds=2)
    g.start_round()
    t = play_inputs(g)
    for _ in range(3):
        g.record_substitution(t, "subject", "carol")
    g.current_round.finished = True
    g.start_round()
    t2 = play_inputs(g)
    assert g.record_substitution(t2, "subject", "carol") is True   # 4回目


# ---- 原則11: 先着判定はサーバーの受信時刻 ----

def test_first_correct_by_server_receive_time_wins():
    """§9-11。クライアントの申告ではなく、渡された受信時刻の順で決まる。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g)
    # bob の回答を先に処理する(受信時刻が早い)
    assert g.submit_answer(t, "bob", correct_picks(g, t, "bob"), received_at=0.8)
    assert g.submit_answer(t, "alice", correct_picks(g, t, "alice"), received_at=0.9)
    assert t.winner == "bob"
    assert g.current_round.points()["bob"] == 1


# ---- 原則12: 補語スロットを追加しない ----

def test_there_are_exactly_three_slots():
    """§9-12。主語・目的語・述語の3つ。"""
    assert SLOTS == ("subject", "object", "predicate")
    assert len(SLOTS) == 3


def test_slots_rotate_every_turn():
    """RULES.md §1.3。3ターンで一巡し、全員が全スロットを担当する。"""
    g = new_game()
    seen = {p: set() for p in P}
    for i in range(3):
        for p, slot in g.assign_slots(i).items():
            seen[p].add(slot)
    for p in P:
        assert seen[p] == set(SLOTS)


# ---- 決着(RULES.md §7) ----

def test_round_ends_at_three_points():
    """3ポイント先取。"""
    g = new_game()
    g.start_round()
    for i in range(3):
        t = play_inputs(g)
        g.submit_answer(t, "alice", correct_picks(g, t, "alice"), received_at=1.0)
        end = g.check_round_end()
    assert end is not None and end.winner == "alice"


def test_ten_turns_is_reachable_only_with_scoreless_turns():
    """§9の根拠: 毎ターン得点が入れば最長7ターンで決着する(3人・3点先取)。"""
    g = new_game()
    g.start_round()
    order = ["alice", "bob", "carol"] * 3
    for i, p in enumerate(order):
        t = play_inputs(g)
        g.submit_answer(t, p, correct_picks(g, t, p), received_at=1.0)
        if g.check_round_end():
            break
    assert g.current_round.finished
    assert g.current_round.counted_turns() == 7


def test_max_turns_then_most_points_wins():
    """10ターンで決着しない場合は最多得点者【現行案】。"""
    g = new_game()
    g.start_round()
    for i in range(10):
        t = play_inputs(g)
        if i < 2:
            g.submit_answer(t, "alice", correct_picks(g, t, "alice"), received_at=1.0)
        else:
            g.finish_turn_by_timeout(t)
        end = g.check_round_end()
    assert end is not None and end.winner == "alice"


def test_max_turns_with_a_tie_has_no_winner():
    """同点なら勝者なしの引き分け【現行案】。"""
    g = new_game()
    g.start_round()
    for i in range(10):
        t = play_inputs(g)
        if i < 2:
            g.submit_answer(t, "alice", correct_picks(g, t, "alice"), received_at=1.0)
        elif i < 4:
            g.submit_answer(t, "bob", correct_picks(g, t, "bob"), received_at=1.0)
        else:
            g.finish_turn_by_timeout(t)
        end = g.check_round_end()
    assert end is not None and end.winner is None
    assert g.round_wins["alice"] == 0 and g.round_wins["bob"] == 0


def test_game_winner_is_most_round_wins_then_total_points():
    """RULES.md §7.2。"""
    g = new_game(rounds=2)
    for winner in ("alice", "bob"):
        g.start_round()
        for _ in range(3):
            t = play_inputs(g)
            g.submit_answer(t, winner, correct_picks(g, t, winner), received_at=1.0)
            g.check_round_end()
    res = g.result()
    assert sorted(res["winners"]) == ["alice", "bob"]      # 勝利数も得点も同じ=同率


# ---- 構成の検査 ----

def test_group_size_and_round_count_are_enforced():
    with pytest.raises(RuleError):
        Game(["a", "b"], 1)
    with pytest.raises(RuleError):
        Game(P, 6)


def test_hint_does_not_collide_with_a_frame_switch():
    """§9-9(RULES.md §2.2)。ヒント5.5秒と切り替えが重ならないこと。"""
    t = Timing()
    assert t.hint_gap_seconds() >= 0.4
    assert max(t.frame_times_seconds) <= t.last_frame_by_seconds


def test_swapped_answer_is_rejected_by_default_and_accepted_when_configured():
    """計画書 §8-14 / RULES.md §11。E2 の結果で切り替える設定。"""
    for accept in (False, True):
        g = Game(P, 1, Config(scoring=Scoring(accept_swapped_answer=accept)))
        g.start_round()
        t = play_inputs(g, {"subject": "犬", "object": "猫", "predicate": "追う"})
        answerer = next(p for p in P if t.slot_of(p) == "predicate")
        swapped = {"subject": "猫", "object": "犬"}
        assert g.submit_answer(t, answerer, swapped, received_at=1.0) is accept


# ---- 通信で必ず起きること ----

def test_answer_arriving_just_after_the_winner_is_not_an_error():
    """先着が決まる直前に送られ、わずかに遅れて届いた回答。

    通信では普通に起きるので、例外にしない。得点は入らず、失格にもしない。
    """
    g = new_game()
    g.start_round()
    t = play_inputs(g)
    g.submit_answer(t, "alice", correct_picks(g, t, "alice"), received_at=1.0)
    late = g.submit_answer(t, "bob", correct_picks(g, t, "bob"), received_at=1.01)
    assert late is True                       # 内容としては正解
    assert g.current_round.points()["bob"] == 0
    assert "bob" not in t.disqualified        # 失格にもしない
    assert t.winner == "alice"


def test_nobody_can_answer_a_void_turn():
    """無効になったターンは誰も回答できない(RULES.md §3.3)。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g, timed_out={"bob", "carol"})
    with pytest.raises(RuleError):
        g.submit_answer(t, "alice", correct_picks(g, t, "alice"), received_at=1.0)


def test_timed_out_player_cannot_answer():
    """§9-6。時間切れの人はそのターン回答できない。"""
    g = new_game()
    g.start_round()
    t = play_inputs(g, timed_out={"carol"})
    with pytest.raises(RuleError):
        g.submit_answer(t, "carol", correct_picks(g, t, "carol"), received_at=1.0)
