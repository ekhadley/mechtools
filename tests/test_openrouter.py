import asyncio
import json

import httpx
import pytest

from mechtools import openrouter as orr
from mechtools.openrouter import *

CHAT_BODY = {"id": "x", "provider": "DeepInfra", "model": "deepseek/deepseek-v4-flash", "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi", "reasoning": "hmm"}}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 7, "cost": 0.001, "completion_tokens_details": {"reasoning_tokens": 3}}}
RAW_BODY = {"id": "x", "provider": "Chutes", "model": "qwen/qwen3.6-27b", "choices": [{"finish_reason": "length", "text": " 391", "reasoning": None}], "usage": {"prompt_tokens": 20, "completion_tokens": 12, "cost": 0.002}}
JSON = {"content-type": "application/json"}

def test_flat():
    assert flat(CHAT_BODY) == {"text": "hi", "reasoning": "hmm", "finish_reason": "stop", "provider": "DeepInfra", "model": "deepseek/deepseek-v4-flash", "prompt_tokens": 5, "completion_tokens": 7, "reasoning_tokens": 3, "cost": 0.001}
    assert flat(RAW_BODY) == {"text": " 391", "reasoning": None, "finish_reason": "length", "provider": "Chutes", "model": "qwen/qwen3.6-27b", "prompt_tokens": 20, "completion_tokens": 12, "reasoning_tokens": None, "cost": 0.002}
    assert flat({**CHAT_BODY, "choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": None, "reasoning": "hmm"}}]})["text"] is None  # a trace that hit max_tokens before answering

def test_verdict():
    assert orr._verdict(200, CHAT_BODY, "") == (None, "")
    assert orr._verdict(200, {"error": {"code": 429, "message": "rate limited"}}, "")[0] == "envelope 429"
    assert orr._verdict(200, {"error": {"code": 502, "message": "upstream"}, "choices": None}, "")[0] == "envelope 502"
    assert orr._verdict(200, {"choices": [{"finish_reason": "error", "text": ""}]}, "")[0] == "finish error"
    for usage in (None, {}, {"prompt_tokens": 1}, {"completion_tokens": 2}, {"completion_tokens": 2, "cost": None}):
        assert orr._verdict(200, {**RAW_BODY, "usage": usage}, "")[0] == "no usage"
    assert orr._verdict(200, {**RAW_BODY, "usage": {"completion_tokens": 2, "cost": 0}}, "") == (None, "")  # free models cost 0, which is fine
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
    asyncio.run(chat("hi", "m", provider={"order": ["a", "b"]}, reasoning=False))
    asyncio.run(complete("x", "m", stop=["a", "b"], max_tokens=5, seed=1))
    assert seen[0] == ("/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8192, "temperature": 1.0, "transforms": [], "usage": {"include": True}, "top_k": 0, "provider": {"only": ["deepinfra"], "allow_fallbacks": False}, "reasoning": {"enabled": True}})
    assert seen[1][1]["reasoning"] == {"effort": "high"} and "provider" not in seen[1][1]
    assert seen[2] == ("/completions", {"model": "m", "prompt": "<think>", "max_tokens": 8192, "temperature": 1.0, "transforms": [], "usage": {"include": True}, "provider": {"only": ["chutes"], "allow_fallbacks": False}, "stop": ["</think>"]})
    assert seen[3][1]["reasoning"] == {"enabled": False} and seen[3][1]["provider"] == {"order": ["a", "b"]}  # False is a value, not an absence; a dict pins as given
    assert seen[4][1]["stop"] == ["a", "b"] and seen[4][1]["max_tokens"] == 5 and seen[4][1]["seed"] == 1

def test_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OTHER=1\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match=f"OPENROUTER_API_KEY is not set: {tmp_path / '.env'} was loaded"):
        endpoints("m")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        asyncio.run(complete("x", "m"))

@pytest.fixture
def post(monkeypatch):
    """complete() against a MockTransport serving the given (status, headers, content) responses in order, the last one repeating; the backoff sleep is stubbed and its arguments collected. Returns (body, request payloads, sleeps) or raises."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    sleeps = []
    async def fake_sleep(s): sleeps.append(s)
    monkeypatch.setattr(orr, "_sleep", fake_sleep)
    def run(responses, attempts=3, **kw):
        payloads = []
        def handler(request):
            r = responses[min(len(payloads), len(responses) - 1)]
            payloads.append(json.loads(request.content))
            if isinstance(r, Exception):
                raise r
            status, headers, content = r
            return httpx.Response(status, headers=headers, content=content)
        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                return await complete("p", "m", attempts=attempts, client=c, **kw)
        orr._stats = Stats()
        return asyncio.run(go()), payloads, sleeps
    return run

def test_post_retries_with_backoff(post):
    body, payloads, sleeps = post([(429, JSON, json.dumps({"error": {"code": 429, "message": "slow"}})), (200, JSON, json.dumps(RAW_BODY))])
    assert body == RAW_BODY and len(payloads) == 2 and sleeps == [1] and payloads[0]["prompt"] == "p"
    assert orr._stats.errors == {"429": 1} and orr._stats.ok == 1 and orr._stats.open == {} and orr._stats.cost == 0.002 and orr._stats.toks == 12 and orr._stats.finish == {"length": 1}
    with pytest.raises(RequestFailed) as e:
        post([(503, {"content-type": "text/html"}, "<html>bad</html>")], attempts=4)
    assert e.value.why == "503" and "<html>bad</html>" in str(e.value) and sleeps[1:] == [1, 3, 7] and orr._stats.errors == {"503": 4} and orr._stats.open == {}

@pytest.mark.parametrize("first,why", [
    ((200, JSON, "{not json"), "bad json"),
    ((200, JSON, json.dumps({k: v for k, v in RAW_BODY.items() if k != "usage"})), "no usage"),
    ((200, JSON, json.dumps({**RAW_BODY, "usage": {"prompt_tokens": 1, "completion_tokens": 2}})), "no usage"),
    ((200, JSON, json.dumps({**RAW_BODY, "usage": {"prompt_tokens": 1, "completion_tokens": 2, "cost": None}})), "no usage"),
    ((200, JSON, json.dumps({"error": {"code": 502, "message": "upstream"}, "choices": None})), "envelope 502"),
    ((200, JSON, json.dumps({**RAW_BODY, "choices": [{"finish_reason": "error", "text": ""}]})), "finish error"),
    ((200, {"content-type": "text/html"}, "<html>captive portal</html>"), "envelope None"),
    ((503, JSON, "{truncated"), "503"),
    (httpx.ConnectError("refused"), "ConnectError"),
    (httpx.ReadTimeout("slow"), "timeout"),
])
def test_post_retries_malformed_responses(post, first, why):
    """Each of these used to escape _post uncaught (killing a whole batch) or is a retry cause; all must be retried and counted."""
    body, payloads, sleeps = post([first, (200, JSON, json.dumps(RAW_BODY))])
    assert body == RAW_BODY and len(payloads) == 2 and sleeps == [1] and orr._stats.errors == {why: 1} and orr._stats.ok == 1

def test_post_hard_failures(post):
    for status in (401, 402):
        with pytest.raises(RuntimeError, match=f"{status}: no"):
            post([(status, JSON, json.dumps({"error": {"code": status, "message": "no"}}))])
    with pytest.raises(RequestFailed) as e:
        post([(400, JSON, json.dumps({"error": {"code": 400, "message": "bad", "metadata": {"provider_name": "P", "raw": "x"}}}))])
    assert e.value.why == "400" and str(e.value) == "400: bad | P | x" and orr._stats.errors == {} and orr._stats.open == {}

def test_batch_stops_on_auth_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    def handler(request):
        return httpx.Response(401, headers=JSON, content=json.dumps({"error": {"code": 401, "message": "no key"}}))
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await gather_bar([complete("p", "m", client=c) for _ in range(3)], desc="t")
    with pytest.raises(ExceptionGroup) as e:
        asyncio.run(go())
    assert all(isinstance(x, RuntimeError) for x in e.value.exceptions)

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
    assert asyncio.run(gather_bar([], desc="t")) == []

def test_gather_bar_bounds_concurrency():
    live, peak = 0, 0
    async def c():
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return live
    assert len(asyncio.run(gather_bar([c() for _ in range(20)], concurrency=4, desc="t"))) == 20 and peak == 4

def test_endpoints(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen = {}
    def fake_get(url, headers, timeout):
        seen["url"], seen["auth"] = url, headers["Authorization"]
        return httpx.Response(200, json={"data": {"endpoints": [{"tag": "chutes/fp8", "quantization": "fp8"}, {"tag": "phala", "quantization": None}]}}, request=httpx.Request("GET", url))
    monkeypatch.setattr(orr.httpx, "get", fake_get)
    eps = endpoints("qwen/qwen3.6-27b")
    assert [e["provider"] for e in eps] == ["chutes", "phala"] and eps[0]["quantization"] == "fp8" and seen["url"].endswith("/models/qwen/qwen3.6-27b/endpoints") and seen["auth"] == "Bearer test-key"
    monkeypatch.setattr(orr.httpx, "get", lambda url, headers, timeout: httpx.Response(404, request=httpx.Request("GET", url)))
    with pytest.raises(httpx.HTTPStatusError):
        endpoints("nope")
