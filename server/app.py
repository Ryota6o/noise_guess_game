"""ゲームサーバー。FastAPI + WebSocket。

計画書 §4.1、§4.3。公平性のルールは RULES.md §8。
**ルールの判断は rules.py がすべて持つ**。ここは出来事を伝えて結果を配るだけ。

守っていること(dq-realtime-sync スキル):

- 先着・正誤の判定は**サーバー側で受信時刻を基準**に行う。
- フレームは**全枚数を作り、事前チェックを通してから**配信する。
- 回答開始の時刻を**サーバー時刻で配り**、各端末がその時刻に合わせて表示する。
  サーバーから「今切り替えろ」と都度送らない(通信の遅い人だけ不利になる)。
- **時間切れの人には画像と選択肢を送らない**。クライアントで隠すのではなく、送らない。
- 入力は**全員が入力したら即終了**。10秒を待たない。

試作では1グループずつの利用を前提にする(計画書 §8-15)。
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT, Config          # noqa: E402
from rules import SLOTS, SLOT_LABELS, Game, RuleError, TurnOutcome  # noqa: E402

# 回答開始までの先読み猶予。フレームを各端末に読み込ませてから一斉に始める。
PRELOAD_GRACE_SECONDS = 0.8


@dataclass
class Player:
    pid: str
    name: str
    ws: WebSocket | None = None
    pending_input: str | None = None


@dataclass
class Room:
    code: str
    rounds: int
    config: Config
    preparer: Any
    players: dict[str, Player] = field(default_factory=dict)
    game: Game | None = None
    started: bool = False

    def order(self) -> list[str]:
        return list(self.players)


class Server:
    def __init__(self, preparer: Any, config: Config = DEFAULT) -> None:
        self.config = config
        self.preparer = preparer
        self.rooms: dict[str, Room] = {}

    # ----- 配信 -----

    async def send(self, player: Player, payload: dict) -> None:
        if player.ws is None:
            return
        try:
            await player.ws.send_text(json.dumps(payload, ensure_ascii=False))
        except (WebSocketDisconnect, RuntimeError):
            player.ws = None

    async def broadcast(self, room: Room, payload: dict,
                        only: set[str] | None = None) -> None:
        targets = [p for pid, p in room.players.items()
                   if only is None or pid in only]
        await asyncio.gather(*(self.send(p, payload) for p in targets))

    @staticmethod
    def public_choices(options: list[dict]) -> list[dict]:
        """クライアントに送ってよい形にする。

        **`role` や `en` を送ってはいけない。** `role: "correct"` が入っていると、
        開発者ツールを開くだけで答えが分かる。表示に要るのは日本語の語だけ。
        """
        return [{"ja": o["ja"]} for o in options]

    @staticmethod
    def hint_targets(options: list[dict]) -> list[str]:
        """5.5秒のヒントで伏せる語(RULES.md §4.5)。中くらいのダミーを1つ。

        伏せる対象は**誤答**なので、先に知られても1択減るだけで答えは漏れない。
        各端末がサーバー時刻に合わせて自分で伏せるため、先に渡しておく(§8)。
        """
        return [o["ja"] for o in options if o.get("role") == "mid"][:1]

    @staticmethod
    def now() -> float:
        """サーバーの単調増加時刻。先着判定と時刻配信の基準。"""
        return asyncio.get_running_loop().time()

    # ----- ターンの進行 -----

    async def run_turn(self, room: Room) -> None:
        game = room.game
        assert game is not None
        turn = game.start_turn()
        t = self.config.timing

        # --- 入力フェイズ。全員が入力したら即終了 ---
        for pid, player in room.players.items():
            player.pending_input = None
            slot = turn.slot_of(pid)
            await self.send(player, {
                "type": "input_start", "turn": turn.index, "slot": slot,
                "slot_label": SLOT_LABELS[slot],
                "seconds": t.input_seconds, "server_now": self.now()})

        deadline = self.now() + t.input_seconds
        while self.now() < deadline:
            if all(p.pending_input for p in room.players.values()):
                break
            await asyncio.sleep(0.05)

        timed_out = {pid for pid, p in room.players.items() if not p.pending_input}
        words = {}
        for pid, player in room.players.items():
            slot = turn.slot_of(pid)
            # 時間切れの人の語はシステムが補う(RULES.md §3.3)
            words[slot] = player.pending_input or self._autofill(room, slot)

        outcome = game.submit_inputs(turn, words, timed_out)
        if outcome is TurnOutcome.VOID:
            await self.broadcast(room, {
                "type": "turn_void", "turn": turn.index,
                "reason": "入力の時間切れが2人以上", "server_now": self.now()})
            return

        # --- 生成と事前チェック。全部終わってから配信する ---
        await self.broadcast(room, {"type": "preparing", "turn": turn.index,
                                    "budget": t.prepare_budget_seconds})
        assets = await asyncio.to_thread(
            self.preparer.prepare_turn, words, 1000 + turn.index)

        if assets.failed:
            await self.broadcast(room, {
                "type": "turn_void", "turn": turn.index,
                "reason": "事前チェックを通せなかった", "server_now": self.now()})
            turn.outcome = TurnOutcome.VOID
            return

        # 差し替えが起きた語の入力者に、減点を記録する(RULES.md §6)
        for slot in assets.substituted:
            owner = next(pid for pid in room.players if turn.slot_of(pid) == slot)
            deducted = game.record_substitution(turn, slot, owner)
            await self.send(room.players[owner], {
                "type": "substituted", "slot": slot,
                "deducted": deducted,
                "count": game.substitutions[owner]})
        turn.words = dict(assets.words_ja)

        # --- 配信。時間切れの人には送らない ---
        answer_start_at = self.now() + PRELOAD_GRACE_SECONDS
        for pid, player in room.players.items():
            if pid in timed_out:
                await self.send(player, {
                    "type": "sit_out", "turn": turn.index,
                    "reason": "入力が時間切れだったので、このターンは回答できません"})
                continue
            await self.send(player, {
                "type": "answer_ready", "turn": turn.index,
                "frames": assets.frames_jpeg_b64,
                "frame_times": list(assets.frame_times),
                "choices": {s: self.public_choices(assets.choices[s])
                            for s in turn.answerable_slots(pid)},
                "hint_hide": {s: self.hint_targets(assets.choices[s])
                              for s in turn.answerable_slots(pid)},
                "answer_slots": turn.answerable_slots(pid),
                "server_now": self.now(),
                "answer_start_at": answer_start_at,
                "answer_seconds": t.answer_seconds,
                "hint_at": t.hint_at_seconds})

        # --- 回答フェイズ ---
        end_at = answer_start_at + t.answer_seconds
        while self.now() < end_at and turn.outcome is TurnOutcome.ONGOING:
            await asyncio.sleep(0.02)
        game.finish_turn_by_timeout(turn)

        # --- 結果表示 ---
        await self.broadcast(room, {
            "type": "turn_result", "turn": turn.index,
            "words": assets.words_ja, "prompt_en": assets.prompt_en,
            "winner": turn.winner, "outcome": turn.outcome.value,
            "points": game.current_round.points(),
            "server_now": self.now(), "seconds": t.result_seconds})
        await asyncio.sleep(t.result_seconds)

    def _autofill(self, room: Room, slot: str) -> str:
        """時間切れの人の語を、過去の入力プールから同カテゴリで補う(RULES.md §3.3)。"""
        from gpu import CATEGORY_OF_SLOT
        picker = getattr(self.preparer, "picker", None)
        game = room.game
        if picker is None:
            return f"{slot}_自動"
        pool = [w.ja for w in picker.pool[CATEGORY_OF_SLOT[slot]]]
        usable = [w for w in pool if game is None or game.word_is_reusable(w)]
        return (usable or pool)[0]

    # ----- ゲーム全体 -----

    async def run_game(self, room: Room) -> None:
        game = room.game
        assert game is not None
        for _ in range(room.rounds):
            rnd = game.start_round()
            await self.broadcast(room, {"type": "round_start", "round": rnd.index + 1,
                                        "of": room.rounds})
            while True:
                await self.run_turn(room)
                if game.check_round_end():
                    break
                if rnd.counted_turns() >= self.config.scoring.max_turns_per_round:
                    game.check_round_end()
                    break
            await self.broadcast(room, {
                "type": "round_end", "round": rnd.index + 1,
                "winner": rnd.winner, "points": rnd.points(),
                "round_wins": dict(game.round_wins)})
        await self.broadcast(room, {"type": "game_end", **game.result()})

    # ----- 受信 -----

    async def handle(self, room: Room, player: Player, message: dict) -> None:
        kind = message.get("type")
        if kind == "input":
            word = (message.get("word") or "").strip()
            game = room.game
            if game and not game.word_is_reusable(word):
                # 同じ語は同一ラウンド内で再利用できない(RULES.md §3.1)
                await self.send(player, {"type": "input_rejected",
                                         "reason": "このラウンドでもう使われた語です"})
                return
            player.pending_input = word
            await self.send(player, {"type": "input_accepted", "word": word})

        elif kind == "answer":
            game = room.game
            if game is None or not game.current_round.turns:
                return
            turn = game.current_round.turns[-1]
            received_at = self.now()          # ★ 先着判定はこの時刻で行う
            try:
                correct = game.submit_answer(turn, player.pid,
                                             message.get("picks") or {}, received_at)
            except RuleError as e:
                await self.send(player, {"type": "answer_rejected", "reason": str(e)})
                return
            await self.send(player, {"type": "answer_result", "correct": correct,
                                     "scored": turn.winner == player.pid})


def create_app(preparer: Any | None = None, config: Config = DEFAULT) -> FastAPI:
    if preparer is None:
        from gpu import TurnPreparer
        preparer = TurnPreparer(config)
    server = Server(preparer, config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """初回の生成とVLM呼び出しは極端に遅い。客が来る前に暖めておく。

        実測(2026-10-16): 暖機なしだと1問目の生成が22.3秒、VLMの初回が12.8秒かかり、
        予算(5秒)を大きく超える。暖機後は合計1.19秒。
        """
        if hasattr(preparer, "warmup"):
            print(f"[warmup] {await asyncio.to_thread(preparer.warmup)}", flush=True)
        yield

    app = FastAPI(title="Diffusion Quest", lifespan=lifespan)
    app.state.server = server

    web_dir = Path(__file__).parents[1] / "web"

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.websocket("/ws/{room_code}/{name}")
    async def ws_endpoint(ws: WebSocket, room_code: str, name: str) -> None:
        await ws.accept()
        room = server.rooms.get(room_code)
        if room is None:
            room = Room(code=room_code, rounds=1, config=config, preparer=preparer)
            server.rooms[room_code] = room

        pid = name
        player = room.players.get(pid) or Player(pid=pid, name=name)
        player.ws = ws
        room.players[pid] = player
        await server.broadcast(room, {"type": "lobby",
                                      "players": [p.name for p in room.players.values()]})

        n = config.scoring.players_per_group
        if len(room.players) == n and not room.started:
            room.started = True
            room.game = Game(room.order(), room.rounds, config)
            asyncio.create_task(server.run_game(room))

        try:
            while True:
                raw = await ws.receive_text()
                await server.handle(room, player, json.loads(raw))
        except WebSocketDisconnect:
            player.ws = None

    return app


app_factory = create_app
