import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evalkit.clients import anthropic_client, openai_compatible, with_budget
from evalkit.runner import AbortRun


@pytest.fixture
def server():
    """本地假接口：按路径返回 OpenAI / Anthropic 格式；第一次请求返回 429 以测重试。"""
    state = {"calls": [], "fail_first": True}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            state["calls"].append(
                (self.path, {k.lower(): v for k, v in self.headers.items()}, body)
            )
            if state["fail_first"]:
                state["fail_first"] = False
                self.send_response(429)
                self.end_headers()
                return
            if self.headers.get("authorization") == "Bearer bad":
                self.send_response(401)
                self.end_headers()
                return
            if self.path.endswith("/chat/completions"):
                out = {
                    "choices": [{"message": {"content": '{"score": 5}'}}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
                }
            else:
                out = {
                    "content": [{"type": "text", "text": '{"winner": "A"}'}],
                    "usage": {
                        "input_tokens": 500,
                        "cache_read_input_tokens": 500,
                        "output_tokens": 100,
                    },
                }
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", state
    srv.shutdown()


def test_openai_compatible_with_retry_and_cost(server, monkeypatch):
    url, state = server
    monkeypatch.setattr("evalkit.clients.time.sleep", lambda s: None)
    c = openai_compatible("m", base_url=url + "/v1", api_key="k", price_per_mtok=(1.0, 10.0))
    text, cost = c("问题")
    assert text == '{"score": 5}' and cost == pytest.approx(0.002)
    assert len(state["calls"]) == 2  # 429 后重试
    assert state["calls"][-1][2]["messages"][-1]["content"] == "问题"


def test_anthropic_client(server):
    url, state = server
    state["fail_first"] = False
    c = anthropic_client("m", api_key="k", base_url=url, price_per_mtok=(1.0, 5.0), system="s")
    text, cost = c("p")
    path, headers, body = state["calls"][-1]
    assert path == "/v1/messages" and headers["x-api-key"] == "k" and body["system"] == "s"
    assert text == '{"winner": "A"}' and cost == pytest.approx(0.0015)


def test_auth_failure_and_missing_key_abort(server, monkeypatch):
    url, state = server
    state["fail_first"] = False
    with pytest.raises(AbortRun, match="鉴权"):
        openai_compatible("m", base_url=url, api_key="bad")("p")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AbortRun, match="密钥"):
        openai_compatible("m", base_url=url)("p")


def test_with_budget():
    c = with_budget(lambda p: ("ok", 0.4), max_cost=1.0)
    for _ in range(3):
        c("x")
    assert c.spent() == pytest.approx(1.2)
    with pytest.raises(AbortRun, match="上限"):
        c("x")
