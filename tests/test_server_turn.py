"""サーバーの進行を、実際に WebSocket をつないで確かめる。

GPU は使わない(`FakeTurnPreparer` に差し替える)。確かめるのは次の点。

- 3人がつないだらゲームが始まる(M4 の完了の目安)
- 入力は**全員が入力したら即終了**する(10秒待たない)
- 時間切れの人には**画像と選択肢が配信されない**(RULES.md §8)
- 先着で正解した1人だけに点が入る
- 回答開始時刻がサーバー時刻で配られる
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
from pathlib import Path

import pytest
import uvicorn
import websockets

sys.path.insert(0, str(Path(__file__).parents[1] / "server"))

from app import create_app                               # noqa: E402
from config import Config, Scoring, Timing               # noqa: E402
from fake_gpu import FakeTurnPreparer                    # noqa: E402

# テストを待たせないために短くする。比率は本番と同じ順序関係を保つ。
TEST_CONFIG = Config(timing=Timing(input_seconds=2.0, answer_seconds=3.0,
                                   result_seconds=0.1))
NAMES = ["alice", "bob", "carol"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    def __init__(self, config: Config) -> None:
        self.port = free_port()
        app = create_app(FakeTurnPreparer(config), config)
        self.server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> "ServerThread":
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                return self
            threading.Event().wait(0.05)
        raise RuntimeError("サーバーが起動しない")

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


async def recv_until(ws, kind: str, timeout: float = 15.0) -> dict:
    """指定した種類のメッセージが来るまで読み飛ばす。"""
    async def loop():
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") == kind:
                return msg
    return await asyncio.wait_for(loop(), timeout)


async def play_one_turn(port: int, silent: set[str] = frozenset()) -> dict:
    """1ターン遊んで、結果と各クライアントが受け取ったものを返す。

    `silent` に入れた名前の人は入力しない(時間切れになる)。
    """
    uri = f"ws://127.0.0.1:{port}/ws/testroom/"
    conns = {name: await websockets.connect(uri + name) for name in NAMES}
    received: dict[str, list[dict]] = {n: [] for n in NAMES}
    try:
        starts = {}
        for name, ws in conns.items():
            starts[name] = await recv_until(ws, "input_start")
            if name not in silent:
                await ws.send(json.dumps({"type": "input", "word": f"{name}の語"}))

        ready: dict[str, dict] = {}
        sat_out: set[str] = set()
        for name, ws in conns.items():
            try:
                msg = await asyncio.wait_for(
                    recv_until(ws, "answer_ready", timeout=12.0), timeout=12.0)
                ready[name] = msg
            except asyncio.TimeoutError:
                sat_out.add(name)

        # 最初の1人だけが正解を送る
        if ready:
            first = sorted(ready)[0]
            msg = ready[first]
            # 正解はクライアントに送られないので、FakeTurnPreparer の規則から復元する
            # (ダミーは "<語>_ダミーN" という名前になっている)
            picks = {slot: next(o["ja"] for o in msg["choices"][slot]
                                if "_ダミー" not in o["ja"])
                     for slot in msg["answer_slots"]}
            await conns[first].send(json.dumps({"type": "answer", "picks": picks}))

        result = await recv_until(conns[NAMES[0]], "turn_result")
        return {"starts": starts, "ready": ready, "sat_out": sat_out,
                "result": result, "first": sorted(ready)[0] if ready else None}
    finally:
        for ws in conns.values():
            await ws.close()


def test_three_players_can_play_one_turn():
    """M4 の完了の目安: 3人がつないで1問遊べる。"""
    with ServerThread(TEST_CONFIG) as srv:
        out = asyncio.run(play_one_turn(srv.port))

    # 担当スロットは3人で重ならない
    slots = {name: msg["slot"] for name, msg in out["starts"].items()}
    assert sorted(slots.values()) == ["object", "predicate", "subject"]

    # 回答するのは自分以外の2スロット(RULES.md §4.1)
    for name, msg in out["ready"].items():
        assert len(msg["answer_slots"]) == 2
        assert slots[name] not in msg["answer_slots"]
        assert set(msg["choices"]) == set(msg["answer_slots"])
        assert all(len(v) == 4 for v in msg["choices"].values())

    # 回答開始の時刻がサーバー時刻で配られている(RULES.md §8)
    for msg in out["ready"].values():
        assert msg["answer_start_at"] > msg["server_now"]
        assert msg["hint_at"] == 5.5

    # フレームは全枚数まとめて届く(1枚ずつ流さない)
    for msg in out["ready"].values():
        assert len(msg["frames"]) == len(msg["frame_times"]) == 8

    # 先着1人だけに点が入る
    result = out["result"]
    assert result["winner"] == out["first"]
    points = result["points"]
    assert points[out["first"]] == 1
    assert sum(points.values()) == 1


def test_timed_out_player_is_not_sent_the_image_or_choices():
    """RULES.md §8【確定】。クライアントで隠すのではなく、サーバーが送らない。"""
    with ServerThread(TEST_CONFIG) as srv:
        out = asyncio.run(play_one_turn(srv.port, silent={"carol"}))

    assert "carol" in out["sat_out"], "時間切れの人に配信されてしまっている"
    assert "carol" not in out["ready"]
    assert set(out["ready"]) == {"alice", "bob"}       # 1人の時間切れではターンは成立する


def test_the_answer_is_never_sent_to_the_client_before_the_result():
    """選択肢に正解の印を混ぜない。

    `role: "correct"` を送ると、開発者ツールを開くだけで答えが分かる。
    早押しゲームなので、これは成立を壊す。
    """
    with ServerThread(TEST_CONFIG) as srv:
        out = asyncio.run(play_one_turn(srv.port))

    for name, msg in out["ready"].items():
        blob = json.dumps(msg, ensure_ascii=False)
        assert "correct" not in blob, f"{name} への配信に正解の印が混ざっている"
        assert "role" not in blob and "sim" not in blob
        for options in msg["choices"].values():
            assert all(set(o) == {"ja"} for o in options), "表示に不要な情報が入っている"
        # ヒントで伏せるのは誤答なので、先に渡してよい
        for slot, hidden in msg["hint_hide"].items():
            shown = {o["ja"] for o in msg["choices"][slot]}
            assert set(hidden) <= shown and len(hidden) <= 1
