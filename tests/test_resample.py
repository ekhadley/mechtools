import asyncio
import gc
import itertools
import json
import math
import os
import weakref

import pytest
from conftest import load_tokenizer

import mechtools.resample as rs
from mechtools.openrouter import RequestFailed
from mechtools.resample import *
from mechtools.stats import wilson

PROMPT = "<|im_start|>user\nWhat is 17 * 23?<|im_end|>\n<|im_start|>assistant\n<think>\n"  # 20 Qwen3 tokens
TEXT = "Okay, 17 * 23: 17 * 20 = 340, 17 * 3 = 51, so 340 + 51 = 391. 𐍈 done."  # 54 tokens; the cut at 51 lands between the two byte tokens of 𐍈
RESPONSE_OPEN = "</think>\n\n"

@pytest.fixture(scope="module")
def tok():
    return load_tokenizer("Qwen/Qwen3-0.6B")

def body(rec: dict) -> dict:
    """A raw /completions response body carrying the fields of a flat record."""
    return {"provider": rec["provider"], "model": rec["model"], "choices": [{"finish_reason": rec["finish_reason"], "text": rec["text"], "reasoning": rec["reasoning"]}],
            "usage": {"prompt_tokens": rec["prompt_tokens"], "completion_tokens": rec["completion_tokens"], "cost": rec["cost"], "completion_tokens_details": {"reasoning_tokens": rec["reasoning_tokens"]}}}

def fake_complete(tok, calls: list, wrap: int = 0, wrap_stage2: int = 0, null_text: bool = False):
    """Stands in for openrouter.complete as a passthrough provider (prompt_tokens from the local tokenizer, plus wrap for one that wraps the prompt, or wrap_stage2 for one that wraps only a response_open prompt). Without stop the generation comes split into reasoning and text; with stop, stage 1 returns the reasoning as text (hitting max_tokens from the full text) and the response_open stage returns the response. null_text sends text: null on stage 1 and the split case."""
    async def complete(prompt, model, provider, stop=None, client=None, **kw):
        calls.append((prompt, stop, kw))
        rec = {"finish_reason": "stop", "provider": "Fake", "model": model, "prompt_tokens": len(tok(prompt, add_special_tokens=False)["input_ids"]) + wrap, "completion_tokens": 5, "reasoning_tokens": None, "cost": 0.001}
        if stop is None:
            return body({"text": None if null_text else "**391**", "reasoning": " more thinking", **rec})
        if prompt.endswith(RESPONSE_OPEN):
            return body({"text": "**391**", "reasoning": None, **rec, "prompt_tokens": rec["prompt_tokens"] + wrap_stage2})
        return body({"text": None if null_text else " more thinking", "reasoning": None, **rec, "finish_reason": "length" if prompt.endswith("done.") else "stop"})
    return complete

def test_prefix_grid(tok, tmp_path):
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "r.jsonl"), "m", "p")
    assert res.n_prompt == 20 and len(res.ids) == 54 and res.rollouts == []
    assert res.grid(20) == [0, 20, 40, 54] and res.grid(1) == list(range(55))
    assert res.prefix(0) == "" and res.prefix(54) == TEXT and TEXT.startswith(res.prefix(50))
    assert [t for t in range(55) if res.prefix(t) is None] == [51]

def test_fill_scores(tok, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rs, "complete", fake_complete(tok, calls))
    async def judge(r):
        return {"match": r["t"] >= 40, "why": "x"} if r["t"] != 54 else {"match": None}
    path = str(tmp_path / "r.jsonl")
    res = Resampler(tok, PROMPT, TEXT, path, "m", "p", judge=judge, max_tokens=64)
    recs = asyncio.run(res.fill({0: 2, 40: 3, 51: 5, 54: 1}))
    assert [(r["t"], r["i"]) for r in recs] == [(0, 0), (0, 1), (40, 0), (40, 1), (40, 2), (54, 0)]  # 51 skipped: its prefix does not retokenize
    assert recs[2] == {"t": 40, "i": 0, "reasoning": " more thinking", "response": "**391**", "finish_reason": "stop", "provider": "Fake", "prompt_tokens": 60, "completion_tokens": 5, "cost": 0.001, "raw": [recs[2]["raw"][0]], "match": True, "why": "x"} and recs[2]["raw"][0]["choices"][0]["text"] == "**391**"
    assert calls[2] == (PROMPT + res.prefix(40), None, {"max_tokens": 64})
    assert sorted(json.loads(line)["t"] for line in open(path)) == [0, 0, 40, 40, 40, 54] and res.rollouts == recs
    assert asyncio.run(res.fill({0: 2, 40: 3})) == []  # no deficit
    res2 = Resampler(tok, PROMPT, TEXT, path, "m", "p", judge=judge)  # a new instance loads the file and tops up
    assert [(r["t"], r["i"]) for r in asyncio.run(res2.fill({40: 4}))] == [(40, 3)]
    sc = res2.scores()
    assert [(s["t"], s["n"], s["k"], s["judged"], s["other"]) for s in sc] == [(0, 2, 0, 2, 0), (40, 4, 4, 4, 0), (54, 1, 0, 0, 1)]
    assert sc[0]["p"] == 0.0 and sc[1]["p"] == 1.0 and sc[1]["ci"] == wilson(4, 4) and math.isnan(sc[2]["p"])
    assert sc[0]["token"] == "" and sc[1]["token"] == tok.decode(res.ids[39:40]) and sc[2]["token"] == "."
    assert len(resample_curve(sc, return_fig=True).data) == 2

def test_two_stage(tok, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rs, "complete", fake_complete(tok, calls))
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "r.jsonl"), "m", "p", stop="</think>", response_open=RESPONSE_OPEN)
    short, full = asyncio.run(res.fill({10: 1, 54: 1}))
    assert len(short.pop("raw")) == 2 and len(full.pop("raw")) == 1
    assert short == {"t": 10, "i": 0, "reasoning": " more thinking", "response": "**391**", "finish_reason": "stop", "provider": "Fake", "prompt_tokens": 30, "completion_tokens": 10, "cost": 0.002}
    assert full == {"t": 54, "i": 0, "reasoning": " more thinking", "response": "", "finish_reason": "length", "provider": "Fake", "prompt_tokens": 74, "completion_tokens": 5, "cost": 0.001}  # stage 1 hit max_tokens: no stage 2
    assert [c for c in calls if c[0].endswith(RESPONSE_OPEN)] == [(PROMPT + res.prefix(10) + " more thinking" + RESPONSE_OPEN, "</think>", {})] and len(calls) == 3

def test_wrapping_provider(tok, tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "complete", fake_complete(tok, [], wrap=4))
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "r.jsonl"), "m", "p")
    with pytest.raises(ExceptionGroup) as e:
        asyncio.run(res.fill({0: 1, 8: 1}))
    assert isinstance(e.value.exceptions[0], RuntimeError) and "counted 24 prompt tokens, expected 20" in str(e.value.exceptions[0])
    assert res.rollouts == [] and not os.path.exists(res.path)
    monkeypatch.setattr(rs, "complete", fake_complete(tok, [], wrap_stage2=4))  # the second call of a two-stage rollout is checked too
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "r.jsonl"), "m", "p", stop="</think>", response_open=RESPONSE_OPEN)
    with pytest.raises(ExceptionGroup) as e:
        asyncio.run(res.fill({10: 1}))
    assert "t=10 stage 2: Fake counted" in str(e.value.exceptions[0]) and res.rollouts == []

def test_null_text(tok, tmp_path, monkeypatch):
    """A provider that sends text: null (reasoning only, or nothing) yields empty strings, not a crash."""
    monkeypatch.setattr(rs, "complete", fake_complete(tok, [], null_text=True))
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "r.jsonl"), "m", "p", stop="</think>", response_open=RESPONSE_OPEN)
    rec, = asyncio.run(res.fill({0: 1}))
    assert rec["reasoning"] == "" and rec["response"] == "**391**" and len(rec["raw"]) == 2
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "s.jsonl"), "m", "p")
    rec, = asyncio.run(res.fill({0: 1}))
    assert rec["response"] == "" and rec["reasoning"] == " more thinking"

def test_no_leak(tok, tmp_path):
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "r.jsonl"), "m", "p")
    assert res.prefix(3) == res.prefix(3)
    ref = weakref.ref(res)
    del res
    gc.collect()
    assert ref() is None  # the prefix cache is per instance, not a class-level functools.cache holding every instance

def test_judge_failure_keeps_rollout(tok, tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "complete", fake_complete(tok, []))
    calls = itertools.count()
    async def judge(r):
        if next(calls) == 0:
            raise RequestFailed("429", "judge throttled")
        return {"match": True}
    path = str(tmp_path / "r.jsonl")
    res = Resampler(tok, PROMPT, TEXT, path, "m", "p", judge=judge)
    a, b = asyncio.run(res.fill({0: 2}))
    assert a["judge_error"] == "RequestFailed: 429: judge throttled" and "match" not in a and b["match"] is True  # the paid rollout is saved, without a verdict
    assert [json.loads(l).get("judge_error") for l in open(path)] == [a["judge_error"], None]
    assert asyncio.run(res.fill({0: 2})) == []  # it counts toward the deficit
    sc, = res.scores()
    assert (sc["n"], sc["judged"], sc["other"]) == (2, 1, 1)
    judged, = asyncio.run(res.judge_pending())
    assert judged is a and "judge_error" not in a and a["match"] is True and res.scores()[0]["judged"] == 2
    assert [json.loads(l).get("match") for l in open(path)] == [True, True] and asyncio.run(res.judge_pending()) == []  # the file is rewritten
    async def broken(r):
        raise TypeError("bug in the judge")
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "b.jsonl"), "m", "p", judge=broken)
    with pytest.raises(ExceptionGroup) as e:
        asyncio.run(res.fill({0: 1}))
    assert isinstance(e.value.exceptions[0], TypeError) and res.rollouts[0]["judge_error"] == "TypeError: bug in the judge" and json.loads(open(res.path).read())["judge_error"]

def test_unique_i_after_failure(tok, tmp_path, monkeypatch):
    calls = []
    passthrough = fake_complete(tok, calls)
    async def flaky(prompt, model, provider, stop=None, client=None, **kw):
        if not calls:
            calls.append(None)
            raise RequestFailed("429", "throttled")
        return await passthrough(prompt, model, provider, stop=stop, client=client, **kw)
    monkeypatch.setattr(rs, "complete", flaky)
    path = str(tmp_path / "r.jsonl")
    res = Resampler(tok, PROMPT, TEXT, path, "m", "p")
    first = asyncio.run(res.fill({0: 2}))
    assert first[0] is None and first[1]["i"] == 1
    assert [r["i"] for r in asyncio.run(res.fill({0: 2}))] == [2]  # the refill does not reuse a failed rollout's i
    on_disk = [(json.loads(l)["t"], json.loads(l)["i"]) for l in open(path)]
    assert on_disk == [(0, 1), (0, 2)]

def test_truncated_file(tok, tmp_path, monkeypatch, capsys):
    path = tmp_path / "r.jsonl"
    good = {"t": 0, "i": 0, "cost": 0.001, "match": True}
    path.write_text(json.dumps(good) + "\n" + '{"t": 1, "i": 0, "co')
    res = Resampler(tok, PROMPT, TEXT, str(path), "m", "p")
    assert res.rollouts == [good] and path.read_text() == json.dumps(good) + "\n" and "truncated last line" in capsys.readouterr().out
    monkeypatch.setattr(rs, "complete", fake_complete(tok, []))
    asyncio.run(res.fill({0: 2}))
    assert [json.loads(l)["i"] for l in open(path)] == [0, 1]  # the append lands on a clean line
    path.write_text('{"bad": \n' + json.dumps(good) + "\n")
    with pytest.raises(ValueError, match="line 1 is not JSON"):
        Resampler(tok, PROMPT, TEXT, str(path), "m", "p")

def fake_probe_complete(tok):
    """Seven endpoints: split (passthrough, continuation in reasoning), merged (passthrough, continuation in text with a special token left in), wrapped (+4 tokens, +5 on the string ending in =), sysprompt (+83), rejected (no raw endpoint), throttled (429 on every attempt), restart (passthrough but the model starts a fresh CoT)."""
    async def complete(prompt, model, provider, max_tokens=8192, attempts=6, client=None, **kw):
        n = len(tok(prompt, add_special_tokens=False)["input_ids"])
        rec = {"finish_reason": "stop", "provider": provider, "model": model, "prompt_tokens": n, "completion_tokens": 9, "reasoning_tokens": 0, "cost": 0.001}
        if provider in ("rejected", "throttled"):
            raise RequestFailed("404" if provider == "rejected" else "429", "no raw endpoint")
        if provider in ("wrapped", "sysprompt"):
            return body({"text": "The product of 17 and 23 is 391.", "reasoning": None, **rec, "prompt_tokens": n + (4 + prompt.endswith("=") if provider == "wrapped" else 83)})
        if provider == "restart":
            return body({"text": None, "reasoning": "Here's a thinking process:", **rec})
        if provider == "merged":
            return body({"text": " 391.<|im_end|>The answer is 391.", "reasoning": None, **rec})
        return body({"text": "**391**" if max_tokens >= 100 else None, "reasoning": " 391. So the answer", **rec})
    return complete

def test_probe(tok, monkeypatch):
    monkeypatch.setattr(rs, "complete", fake_probe_complete(tok))
    monkeypatch.setattr(rs, "endpoints", lambda model: [{"provider": p, "quantization": "fp8", "pricing": {"prompt": "0.0000003", "completion": "0.000002"}} for p in ["split", "merged", "wrapped", "sysprompt", "rejected", "throttled", "restart"]])
    rows = {r["provider"]: r for r in asyncio.run(probe(tok, "m", lambda msgs: PROMPT, samples=2, long=100, extra=(PROMPT + TEXT,)))}
    assert rows["split"]["verdict"] == "pass" and rows["split"]["offsets"] == [0, 0, 0, 0] and rows["split"]["continues"] == "2/2" and rows["split"]["field"] == "reasoning" and rows["split"]["special"] == []
    n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    assert rows["split"]["finish"] == "stop" and rows["split"]["extra_toks"] == 9 - n(" 391. So the answer") - n("**391**") and rows["split"]["reasoning_toks"] == f"0 vs {n(' 391. So the answer')} local"
    assert rows["merged"]["verdict"] == "pass, merged: needs stop + response_open" and rows["merged"]["field"] == "text" and rows["merged"]["special"] == ["<|im_end|>"]
    assert rows["wrapped"] == {"provider": "wrapped", "quant": "fp8", "$/M in,out": "0.30, 2.00", "offsets": [4, 5, 4, 4], "verdict": "wrapped"}  # the partial-CoT string ends "=", one more token in the wrapper
    assert rows["sysprompt"]["verdict"] == "wrapped + system prompt" and rows["sysprompt"]["offsets"] == [83, 83, 83, 83]
    assert rows["rejected"]["verdict"] == "rejected 404" and "offsets" not in rows["rejected"] and rows["throttled"]["verdict"] == "unreachable 429"
    assert rows["restart"]["verdict"] == "restarts" and rows["restart"]["continues"] == "0/2" and rows["restart"]["sample"] == "Here's a thinking process:"
    assert [r["provider"] for r in asyncio.run(probe(tok, "m", lambda msgs: PROMPT, providers=["split"], samples=1, long=100))] == ["split"]
    monkeypatch.setattr(rs, "endpoints", lambda model: [{"provider": "split"}])  # an endpoint entry without pricing or quantization is still probed
    row, = asyncio.run(probe(tok, "m", lambda msgs: PROMPT, samples=1, long=100))
    assert row["$/M in,out"] == "?, ?" and row["quant"] is None and row["verdict"] == "pass"

def test_sampling_defaults(monkeypatch):
    draws = itertools.count()
    async def complete(prompt, model, provider, max_tokens, client=None, **kw):
        k = next(draws) % (4 if "top_k" in kw else 2)  # truncated to 2 distinct tokens unless top_k is passed explicitly
        return body({"text": f" w{k}", "reasoning": None, "finish_reason": "length", "provider": provider, "model": model, "prompt_tokens": 12, "completion_tokens": 1, "reasoning_tokens": None, "cost": 0.0001})
    monkeypatch.setattr(rs, "complete", complete)
    counts = asyncio.run(sampling_defaults("m", "p", "Once upon a time, there lived a", n=8, concurrency=4))
    assert counts["omitted"] == {" w0": 4, " w1": 4} and counts["explicit"] == {" w0": 2, " w1": 2, " w2": 2, " w3": 2}
