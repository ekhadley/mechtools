import asyncio
import itertools
import json
import math
import os

import pytest
from transformers import AutoTokenizer

import mechtools.resample as rs
from mechtools.openrouter import RequestFailed
from mechtools.resample import *
from mechtools.stats import wilson

PROMPT = "<|im_start|>user\nWhat is 17 * 23?<|im_end|>\n<|im_start|>assistant\n<think>\n"  # 20 Qwen3 tokens
TEXT = "Okay, 17 * 23: 17 * 20 = 340, 17 * 3 = 51, so 340 + 51 = 391. 𐍈 done."  # 54 tokens; the cut at 51 lands between the two byte tokens of 𐍈
RESPONSE_OPEN = "</think>\n\n"

@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

def fake_complete(tok, calls: list, wrap: int = 0):
    """Stands in for openrouter.complete as a passthrough provider (prompt_tokens from the local tokenizer, plus wrap for one that wraps the prompt). Without stop the generation comes split into reasoning and text; with stop, stage 1 returns the reasoning as text (hitting max_tokens from the full text) and the response_open stage returns the response."""
    async def complete(prompt, model, provider, stop=None, client=None, **kw):
        calls.append((prompt, stop, kw))
        rec = {"finish_reason": "stop", "provider": "Fake", "model": model, "prompt_tokens": len(tok(prompt, add_special_tokens=False)["input_ids"]) + wrap, "completion_tokens": 5, "reasoning_tokens": None, "cost": 0.001}
        if stop is None:
            return {"text": "**391**", "reasoning": " more thinking", **rec}
        if prompt.endswith(RESPONSE_OPEN):
            return {"text": "**391**", "reasoning": None, **rec}
        return {"text": " more thinking", "reasoning": None, **rec, "finish_reason": "length" if prompt.endswith("done.") else "stop"}
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
    assert recs[2] == {"t": 40, "i": 0, "reasoning": " more thinking", "response": "**391**", "finish_reason": "stop", "provider": "Fake", "prompt_tokens": 60, "completion_tokens": 5, "cost": 0.001, "match": True, "why": "x"}
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

def fake_probe_complete(tok):
    """Seven endpoints: split (passthrough, continuation in reasoning), merged (passthrough, continuation in text with a special token left in), wrapped (+4 tokens, +5 on the string ending in =), sysprompt (+83), rejected (no raw endpoint), throttled (429 on every attempt), restart (passthrough but the model starts a fresh CoT)."""
    async def complete(prompt, model, provider, max_tokens=8192, attempts=6, client=None, **kw):
        n = len(tok(prompt, add_special_tokens=False)["input_ids"])
        rec = {"finish_reason": "stop", "provider": provider, "model": model, "prompt_tokens": n, "completion_tokens": 9, "reasoning_tokens": 0, "cost": 0.001}
        if provider in ("rejected", "throttled"):
            raise RequestFailed("404" if provider == "rejected" else "429", "no raw endpoint")
        if provider in ("wrapped", "sysprompt"):
            return {"text": "The product of 17 and 23 is 391.", "reasoning": None, **rec, "prompt_tokens": n + (4 + prompt.endswith("=") if provider == "wrapped" else 83)}
        if provider == "restart":
            return {"text": None, "reasoning": "Here's a thinking process:", **rec}
        if provider == "merged":
            return {"text": " 391.<|im_end|>The answer is 391.", "reasoning": None, **rec}
        return {"text": "**391**" if max_tokens >= 100 else None, "reasoning": " 391. So the answer", **rec}
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

def test_sampling_defaults(monkeypatch):
    draws = itertools.count()
    async def complete(prompt, model, provider, max_tokens, client=None, **kw):
        k = next(draws) % (4 if "top_k" in kw else 2)  # truncated to 2 distinct tokens unless top_k is passed explicitly
        return {"text": f" w{k}", "reasoning": None, "finish_reason": "length", "provider": provider, "model": model, "prompt_tokens": 12, "completion_tokens": 1, "reasoning_tokens": None, "cost": 0.0001}
    monkeypatch.setattr(rs, "complete", complete)
    counts = asyncio.run(sampling_defaults("m", "p", "Once upon a time, there lived a", n=8, concurrency=4))
    assert counts["omitted"] == {" w0": 4, " w1": 4} and counts["explicit"] == {" w0": 2, " w1": 2, " w2": 2, " w3": 2}
