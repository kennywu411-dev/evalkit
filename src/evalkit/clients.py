"""裁判模型客户端：只用标准库（urllib）调用 HTTP 接口，返回 (原话, 花费美元)，可直接传给 judge.py。

    from evalkit.clients import anthropic_client, openai_compatible, with_budget
    judge_llm = with_budget(anthropic_client("<模型 id>", price_per_mtok=(1.0, 5.0)), max_cost=2.0)
    judge = PointwiseJudge(judge_llm, rubric="...")

- openai_compatible：OpenAI Chat Completions 格式（各家兼容接口、本地推理服务都可以）；
- anthropic_client：Anthropic Messages API；
- price_per_mtok：(输入, 输出) 每百万 token 的美元价格；不给则花费记 0，花费上限也就不起作用；
- with_budget：累计花费达到上限后抛 AbortRun，评测运行会停下并提示续跑。
限流和服务端错误（429、5xx）按指数退避重试。
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from evalkit.runner import AbortRun

RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}


def _post(url: str, headers: dict, body: dict, timeout: float, retries: int) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url, data=data, headers={"content-type": "application/json", **headers}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise AbortRun(f"鉴权失败（HTTP {e.code}），检查密钥") from e
            if e.code not in RETRY_STATUS or attempt == retries:
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries:
                raise RuntimeError(f"请求失败：{e}") from e
        time.sleep(min(30, 2**attempt))
    raise RuntimeError("unreachable")


def _cost(price: tuple[float, float] | None, tokens_in: int, tokens_out: int) -> float:
    if not price:
        return 0.0
    return (tokens_in * price[0] + tokens_out * price[1]) / 1e6


def _key(api_key: str | None, env: str) -> str:
    key = api_key or os.environ.get(env)
    if not key:
        raise AbortRun(f"缺少 API 密钥：传 api_key 或设置环境变量 {env}")
    return key


def openai_compatible(
    model: str,
    base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    price_per_mtok: tuple[float, float] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    system: str | None = None,
    timeout: float = 120,
    retries: int = 4,
) -> Callable[[str], tuple[str, float]]:
    url = base_url.rstrip("/") + "/chat/completions"

    def client(prompt: str) -> tuple[str, float]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        out = _post(
            url,
            {"authorization": f"Bearer {_key(api_key, api_key_env)}"},
            {
                "model": model,
                "messages": msgs,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout,
            retries,
        )
        u = out.get("usage") or {}
        text = out["choices"][0]["message"].get("content") or ""
        return text, _cost(price_per_mtok, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))

    return client


def anthropic_client(
    model: str,
    api_key: str | None = None,
    api_key_env: str = "ANTHROPIC_API_KEY",
    price_per_mtok: tuple[float, float] | None = None,
    max_tokens: int = 1024,
    system: str | None = None,
    base_url: str = "https://api.anthropic.com",
    timeout: float = 120,
    retries: int = 4,
) -> Callable[[str], tuple[str, float]]:
    url = base_url.rstrip("/") + "/v1/messages"

    def client(prompt: str) -> tuple[str, float]:
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        out = _post(
            url,
            {"x-api-key": _key(api_key, api_key_env), "anthropic-version": "2023-06-01"},
            body,
            timeout,
            retries,
        )
        text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
        u = out.get("usage") or {}
        tokens_in = (
            u.get("input_tokens", 0)
            + u.get("cache_read_input_tokens", 0)
            + u.get("cache_creation_input_tokens", 0)
        )
        return text, _cost(price_per_mtok, tokens_in, u.get("output_tokens", 0))

    return client


def with_budget(client: Callable, max_cost: float) -> Callable[[str], tuple[str, float]]:
    """给客户端加累计花费上限（线程安全）。已花费可从 wrapped.spent() 读取。"""
    lock = threading.Lock()
    spent = [0.0]

    def wrapped(prompt: str) -> tuple[str, float]:
        with lock:
            if spent[0] >= max_cost:
                raise AbortRun(f"裁判花费已达上限 ${max_cost:.2f}")
        out = client(prompt)
        text, cost = out if isinstance(out, tuple) else (out, 0.0)
        with lock:
            spent[0] += cost
        return text, cost

    wrapped.spent = lambda: spent[0]  # type: ignore[attr-defined]
    return wrapped
