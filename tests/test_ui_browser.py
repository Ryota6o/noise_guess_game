"""スマホUIを実際のブラウザで動かす。M5 の確認。

GPU は使わない(`FakeTurnPreparer`)。ブラウザ3枚で1ターン遊び、次を確かめる。

- 3人がロビーから入力画面に進む
- 入力画面に自分の担当スロット(だれが/なにを/どうした)が出る
- 回答画面でフレームと4択が出て、**自分以外の2スロット分だけ**表示される
- 選択肢の位置が10秒間動かない(RULES.md §4.6)
- 送信すると結果が出る

Chrome が無い環境では skip する。
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "server"))

playwright_api = pytest.importorskip("playwright.sync_api")

import uvicorn                                   # noqa: E402
from app import create_app                       # noqa: E402
from config import Config, Timing                # noqa: E402
from fake_gpu import FakeTurnPreparer            # noqa: E402

NAMES = ["alice", "bob", "carol"]
SLOT_LABELS = {"だれが", "なにを", "どうした"}
UI_CONFIG = Config(timing=Timing(input_seconds=6.0, answer_seconds=8.0,
                                 result_seconds=0.5))


@pytest.fixture(scope="module")
def server_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(
        create_app(FakeTurnPreparer(UI_CONFIG), UI_CONFIG),
        host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.05)
    yield port
    srv.should_exit = True
    thread.join(timeout=5)


def test_three_browsers_play_a_turn(server_port, tmp_path):
    from playwright.sync_api import sync_playwright

    shots = Path(__file__).parents[1] / "web" / "screenshots"
    shots.mkdir(exist_ok=True)

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome")
        except Exception as e:                      # noqa: BLE001
            pytest.skip(f"Chrome が使えない: {e}")

        # iPhone くらいの画面で確認する
        context = browser.new_context(viewport={"width": 390, "height": 844},
                                      device_scale_factor=2, is_mobile=True)
        pages = {}
        for name in NAMES:
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{server_port}/?room=ui&name={name}")
            pages[name] = page

        # 3人そろうと入力画面になる
        labels = {}
        for name, page in pages.items():
            page.wait_for_selector("#input.on", timeout=15000)
            labels[name] = page.inner_text("#slotLabel")
        assert set(labels.values()) == SLOT_LABELS, f"担当が重なっている: {labels}"

        pages["alice"].screenshot(path=shots / "1_input.png")

        for name, page in pages.items():
            page.fill("#word", f"{name}の語")
            page.click("#sendWord")

        # 回答画面
        for page in pages.values():
            page.wait_for_selector("#answer.on", timeout=20000)

        answerer = pages["alice"]
        answerer.wait_for_selector(".choices button", timeout=10000)
        groups = answerer.query_selector_all("#slots > .panel")
        assert len(groups) == 2, "自分以外の2スロット分だけ出るはず"
        for g in groups:
            assert len(g.query_selector_all("button")) == 4

        # 選択肢の並びは固定(RULES.md §4.6)
        before = answerer.eval_on_selector_all(
            ".choices button", "els => els.map(e => e.textContent)")
        time.sleep(2.0)
        after = answerer.eval_on_selector_all(
            ".choices button", "els => els.map(e => e.textContent)")
        assert before == after, "選択肢の位置が動いている"

        answerer.screenshot(path=shots / "2_answer.png")

        # 正解を選んで送信(FakeTurnPreparer のダミーは "_ダミー" を含む)
        for g in groups:
            for b in g.query_selector_all("button"):
                if "_ダミー" not in b.text_content():
                    b.click()
                    break
        answerer.wait_for_selector("#submit:not([disabled])", timeout=5000)
        answerer.click("#submit")

        answerer.wait_for_selector("#result.on", timeout=20000)
        title = answerer.inner_text("#resultTitle")
        assert "alice" in title, f"結果が出ていない: {title}"
        answerer.screenshot(path=shots / "3_result.png")

        context.close()
        browser.close()
