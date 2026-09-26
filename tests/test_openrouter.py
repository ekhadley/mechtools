import asyncio
import json
import os

import httpx
import pytest

from mechtools import openrouter as orr
from mechtools.openrouter import *

CHAT_BODY = {"id": "x", "provider": "DeepInfra", "model": "deepseek/deepseek-v4-flash", "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi", "reasoning": "hmm"}}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 7, "cost": 0.001, "completion_tokens_details": {"reasoning_tokens": 3}}}
RAW_BODY = {"id": "x", "provider": "Chutes", "model": "qwen/qwen3.6-27b", "choices": [{"finish_reason": "length", "text": " 391", "reasoning": None}], "usage": {"prompt_tokens": 20, "completion_tokens": 12, "cost": 0.002}}

def test_flat():
    assert flat(CHAT_BODY) == {"text": "hi", "reasoning": "hmm", "finish_reason": "stop", "provider": "DeepInfra", "model": "deepseek/deepseek-v4-flash", "prompt_tokens": 5, "completion_tokens": 7, "reasoning_tokens": 3, "cost": 0.001}
    assert flat(RAW_BODY) == {"text": " 391", "reasoning": None, "finish_reason": "length", "provider": "Chutes", "model": "qwen/qwen3.6-27b", "prompt_tokens": 20, "completion_tokens": 12, "reasoning_tokens": None, "cost": 0.002}

def test_verdict():
    assert orr._verdict(200, CHAT_BODY, "") == (None, "")
    assert orr._verdict(200, {"error": {"code": 429, "message": "rate limited"}}, "")[0] == "envelope 429"
    assert orr._verdict(200, {"choices": [{"finish_reason": "error", "text": ""}]}, "")[0] == "finish error"
    assert orr._verdict(429, {"error": {"code": 429, "message": "slow down"}}, "")[0] == "429"
    assert orr._verdict(503, {}, "<html>bad gateway</html>") == ("503", "<html>bad gateway</html>")
    with pytest.raises(RuntimeError):
        orr._verdict(402, {"error": {"code": 402, "message": "insufficient credits"}}, "")
    with pytest.raises(RequestFailed) as e:
        orr._verdict(400, {"error": {"code": 400, "message": "bad model"}}, "")
    assert e.value.why == "400"

def test_payloads(monkeypatch):
    seen = []
    async def fake_post(client, path, payload, attempts, timeout):
        seen.append((path, payload))
        return CHAT_BODY
    monkeypatch.setattr(orr, "_post", fake_post)
    assert asyncio.run(chat("hi", "m", provider="deepinfra", reasoning=True, top_k=0)) is CHAT_BODY
    asyncio.run(chat([{"role": "user", "content": "hi"}], "m", reasoning="high"))
    asyncio.run(complete("<think>", "m", provider="chutes", stop="</think>"))
    assert seen[0] == ("/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8192, "temperature": 1.0, "transforms": [], "usage": {"include": True}, "top_k": 0, "provider": {"only": ["deepinfra"], "allow_fallbacks": False}, "reasoning": {"enabled": True}})
    assert seen[1][1]["reasoning"] == {"effort": "high"} and "provider" not in seen[1][1]
    assert seen[2] == ("/completions", {"model": "m", "prompt": "<think>", "max_tokens": 8192, "temperature": 1.0, "transforms": [], "usage": {"include": True}, "provider": {"only": ["chutes"], "allow_fallbacks": False}, "stop": ["</think>"]})

def test_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OTHER=1\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match=f"OPENROUTER_API_KEY is not set: {tmp_path / '.env'} was loaded"):
        endpoints("m")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        asyncio.run(complete("x", "m"))

def test_gather_bar():
    async def ok(i):
        await asyncio.sleep(0.01 * (3 - i))  # finish out of order
        return i
    async def bad():
        raise RequestFailed("429", "throttled")
    async def bug():
        raise ValueError("not a request problem")
    assert asyncio.run(gather_bar([ok(0), bad(), ok(2), bad()], concurrency=2, desc="t")) == [0, None, 2, None]
    with pytest.raises(ExceptionGroup):
        asyncio.run(gather_bar([ok(0), bug()], desc="t"))
    assert asyncio.run(gather_bar([bug()], desc="t", swallow=(ValueError,))) == [None]

class FakeResponse:
    def __init__(self, status: int, body: dict | None = None, text: str = ""):
        self.status_code, self._body = status, body
        self.text = json.dumps(body) if body is not None else text
        self.headers = {"content-type": "application/json" if body is not None else "text/html"}
    def json(self):
        return self._body

class FakeClient:
    """post answers with the scripted responses in order; an exception in the script is raised instead of returned."""
    def __init__(self, script: list):
        self.script, self.calls = list(script), []
    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(json)
        r = self.script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

@pytest.fixture
def no_backoff(monkeypatch):
    """Replaces the backoff sleep with a recorder and sets a key; returns the list of requested sleeps."""
    sleeps = []
    async def fake_sleep(s):
        sleeps.append(s)
    monkeypatch.setattr(orr.asyncio, "sleep", fake_sleep)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    return sleeps

def test_post_retries_the_retryable_and_counts_each_cause(no_backoff):
    error_finish = {"choices": [{"finish_reason": "error", "text": ""}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}}
    client = FakeClient([FakeResponse(429, {"error": {"code": 429, "message": "slow down"}}), httpx.ReadTimeout("x"), FakeResponse(200, {"error": {"code": 502, "message": "upstream"}}),
                         FakeResponse(200, error_finish), FakeResponse(503, text="<html>bad gateway</html>"), FakeResponse(200, CHAT_BODY)])
    orr._stats = orr.Stats()
    body = asyncio.run(orr._post(client, "/chat/completions", {"model": "m"}, attempts=6, timeout=1))
    assert body is CHAT_BODY and len(client.calls) == 6 and no_backoff == [1, 3, 7, 15, 31]
    s = orr._stats
    assert s.errors == {"429": 1, "timeout": 1, "envelope 502": 1, "finish error": 1, "503": 1} and s.ok == 1 and s.finish == {"stop": 1}
    assert s.cost == 0.001 and s.toks == 7 and s.open == {} and s.sleeping == 0

def test_post_exhausts_rejects_and_stops(no_backoff):
    client = FakeClient([FakeResponse(503, text="down")] * 3)
    with pytest.raises(RequestFailed) as e:
        asyncio.run(orr._post(client, "/completions", {}, attempts=3, timeout=1))
    assert e.value.why == "503" and len(client.calls) == 3 and no_backoff == [1, 3]
    client = FakeClient([FakeResponse(400, {"error": {"code": 400, "message": "bad model"}})])
    with pytest.raises(RequestFailed) as e:
        asyncio.run(orr._post(client, "/completions", {}, attempts=6, timeout=1))
    assert e.value.why == "400" and len(client.calls) == 1  # a 4xx is not retried
    client = FakeClient([FakeResponse(402, {"error": {"code": 402, "message": "insufficient credits"}})])
    with pytest.raises(RuntimeError, match="402"):
        asyncio.run(orr._post(client, "/completions", {}, attempts=6, timeout=1))
    assert orr._stats.open == {}

def test_gather_bar_aborts_a_batch_that_never_succeeds():
    async def bad():
        raise RequestFailed("404", "no such model")
    async def ok():
        return 1
    with pytest.raises(ExceptionGroup) as e:
        asyncio.run(gather_bar([bad() for _ in range(10)], desc="t", abort_after=3))
    assert isinstance(e.value.exceptions[0], RuntimeError) and "3 coroutines failed before any succeeded (404×3)" in str(e.value.exceptions[0])
    assert asyncio.run(gather_bar([bad() for _ in range(4)], desc="t", abort_after=None)) == [None] * 4
    assert asyncio.run(gather_bar([ok()] + [bad() for _ in range(5)], desc="t", abort_after=3)) == [1] + [None] * 5  # after a success, failures are per-request

def test_load_env(monkeypatch, tmp_path, capsys):
    (tmp_path / ".env").write_text("SHADOWED=from_env_file\nSAME=same\nFRESH=new\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHADOWED", "from_shell")
    monkeypatch.setenv("SAME", "same")
    monkeypatch.delenv("FRESH", raising=False)
    assert load_env() == ["SHADOWED"]
    assert os.environ["SHADOWED"] == "from_shell" and os.environ["FRESH"] == "new"
    out = capsys.readouterr().out
    assert "SHADOWED is set in the environment" in out and "SAME" not in out and "FRESH" not in out and "from_shell" not in out and "from_env_file" not in out
