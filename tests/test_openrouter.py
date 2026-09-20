import asyncio

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
