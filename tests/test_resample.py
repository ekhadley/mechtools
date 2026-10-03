import asyncio
import gc
import itertools
import json
import math
import os
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch as t
from conftest import load_tokenizer

import mechtools.resample as rs
from mechtools.openrouter import RequestFailed
from mechtools.plots import DARK, SERIES
from mechtools.resample import *
from mechtools.sampling import eos_ids, sample_rolling
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

def fake_complete(tok, calls: list, wrap: int = 0, null_text: bool = False):
    """Stands in for openrouter.complete as a passthrough provider (prompt_tokens from the local tokenizer, plus wrap for one that wraps the prompt). Without stop the generation comes split into reasoning and text; with stop, stage 1 returns the reasoning as text (hitting max_tokens from the full text) and the response_open stage returns the response. null_text sends text: null on stage 1 and the split case."""
    async def complete(prompt, model, provider, stop=None, client=None, **kw):
        calls.append((prompt, stop, kw))
        rec = {"finish_reason": "stop", "provider": "Fake", "model": model, "prompt_tokens": len(tok(prompt, add_special_tokens=False)["input_ids"]) + wrap, "completion_tokens": 5, "reasoning_tokens": None, "cost": 0.001}
        if stop is None:
            return body({"text": None if null_text else "**391**", "reasoning": " more thinking", **rec})
        if prompt.endswith(RESPONSE_OPEN):
            return body({"text": "**391**", "reasoning": None, **rec})
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
    assert recs[2] == {"t": 40, "i": 0, "reasoning": " more thinking", "response": "**391**", "finish_reason": "stop", "provider": "Fake", "prompt_tokens": 60, "completion_tokens": 5, "cost": 0.001, "cfg": res.cfg, "raw": [recs[2]["raw"][0]], "match": True, "why": "x"} and recs[2]["raw"][0]["choices"][0]["text"] == "**391**"
    assert res.cfg == {"model": "m", "provider": "p", "stop": None, "response_open": "", "kw": {"max_tokens": 64}, "prompt": res.cfg["prompt"], "text": res.cfg["text"]} and len(res.cfg["prompt"]) == 16
    assert calls[2] == (PROMPT + res.prefix(40), None, {"max_tokens": 64})
    assert sorted(json.loads(line)["t"] for line in open(path)) == [0, 0, 40, 40, 40, 54] and res.rollouts == recs and "match" not in json.loads(open(path).readline())
    assert sorted((v["t"], v["i"], v["match"], v.get("why")) for v in map(json.loads, open(res.judged_path))) == [(0, 0, False, "x"), (0, 1, False, "x"), (40, 0, True, "x"), (40, 1, True, "x"), (40, 2, True, "x"), (54, 0, None, None)]
    assert asyncio.run(res.fill({0: 2, 40: 3})) == []  # no deficit
    res2 = Resampler(tok, PROMPT, TEXT, path, "m", "p", judge=judge, max_tokens=64)  # a new instance with the same configuration loads the file and tops up
    assert [(r["t"], r["i"]) for r in asyncio.run(res2.fill({40: 4}))] == [(40, 3)]
    sc = res2.scores()
    assert [(s["t"], s["n"], s["k"], s["judged"], s["other"]) for s in sc] == [(0, 2, 0, 2, 0), (40, 4, 4, 4, 0), (54, 1, 0, 0, 1)] and all(s["direct"] == s["n"] for s in sc)  # the fake never continues the text, so no reuse
    assert sc[0]["p"] == 0.0 and sc[1]["p"] == 1.0 and sc[1]["ci"] == wilson(4, 4) and math.isnan(sc[2]["p"])
    assert sc[0]["token"] == "" and sc[1]["token"] == tok.decode(res.ids[39:40]) and sc[2]["token"] == "."
    assert len(resample_curve(sc, return_fig=True).data) == 2

def test_two_stage(tok, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rs, "complete", fake_complete(tok, calls))
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "r.jsonl"), "m", "p", stop="</think>", response_open=RESPONSE_OPEN)
    short, full = asyncio.run(res.fill({10: 1, 54: 1}))
    assert len(short.pop("raw")) == 2 and len(full.pop("raw")) == 1 and short.pop("cfg") == res.cfg == full.pop("cfg")
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

def fake_probe_complete(tok):
    """Eight endpoints: split (passthrough, continuation in reasoning), merged (passthrough, continuation in text with a special token left in), wrapped (+4 tokens, +5 on the string ending in =), sysprompt (+83), rejected (no raw endpoint), throttled (429 on every attempt), restart (passthrough but the model starts a fresh CoT), hidden (passthrough, but 40 completion tokens never appear in the returned text).
    completion_tokens is the local count of the returned fields plus 2, like a real split provider."""
    n = lambda s: len(tok(s or "", add_special_tokens=False)["input_ids"])
    async def complete(prompt, model, provider, max_tokens=8192, attempts=6, client=None, **kw):
        if provider in ("rejected", "throttled"):
            raise RequestFailed("404" if provider == "rejected" else "429", "no raw endpoint")
        fields = ({"text": "The product of 17 and 23 is 391.", "reasoning": None} if provider in ("wrapped", "sysprompt") else {"text": None, "reasoning": "Here's a thinking process:"} if provider == "restart"
                  else {"text": " 391.<|im_end|>The answer is 391.", "reasoning": None} if provider == "merged" else {"text": "**391**" if max_tokens >= 100 else None, "reasoning": " 391. So the answer"})
        prompt_tokens = n(prompt) + (4 + prompt.endswith("=") if provider == "wrapped" else 83 if provider == "sysprompt" else 0)
        return body({**fields, "finish_reason": "stop", "provider": provider, "model": model, "prompt_tokens": prompt_tokens, "completion_tokens": n(fields["reasoning"]) + n(fields["text"]) + (42 if provider == "hidden" else 2), "reasoning_tokens": 0, "cost": 0.001})
    return complete

def test_probe(tok, monkeypatch):
    monkeypatch.setattr(rs, "complete", fake_probe_complete(tok))
    monkeypatch.setattr(rs, "endpoints", lambda model: [{"provider": p, "quantization": "fp8", "pricing": {"prompt": "0.0000003", "completion": "0.000002"}} for p in ["split", "merged", "wrapped", "sysprompt", "rejected", "throttled", "restart", "hidden"]])
    rows = {r["provider"]: r for r in asyncio.run(probe(tok, "m", lambda msgs: PROMPT, samples=2, long=100, extra=(PROMPT + TEXT,)))}
    assert rows["split"]["verdict"] == "pass" and rows["split"]["offsets"] == [0, 0, 0, 0] and rows["split"]["continues"] == "2/2" and rows["split"]["field"] == "reasoning" and rows["split"]["special"] == []
    n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    assert rows["split"]["finish"] == "stop" and rows["split"]["extra_toks"] == 2 and rows["split"]["reasoning_toks"] == f"0 vs {n(' 391. So the answer')} local"
    assert rows["merged"]["verdict"] == "pass, merged: needs stop + response_open, leaks special tokens" and rows["merged"]["field"] == "text" and rows["merged"]["special"] == ["<|im_end|>"]
    assert rows["hidden"]["verdict"] == "pass, extra_toks off" and rows["hidden"]["extra_toks"] == 42
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

def test_load_checks_the_file_against_the_instance(tok, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rs, "complete", fake_complete(tok, []))
    path = str(tmp_path / "r.jsonl")
    res = Resampler(tok, PROMPT, TEXT, path, "m", "p", max_tokens=64)
    asyncio.run(res.fill({0: 1, 40: 1}))
    assert len(Resampler(tok, PROMPT, TEXT, path, "m", "p", max_tokens=64).rollouts) == 2  # the same configuration loads
    with pytest.raises(ValueError, match="kw differ.*64"):
        Resampler(tok, PROMPT, TEXT, path, "m", "p", max_tokens=128)
    with pytest.raises(ValueError, match="model differ"):
        Resampler(tok, PROMPT, TEXT, path, "other", "p", max_tokens=64)
    with pytest.raises(ValueError, match="text differ"):
        Resampler(tok, PROMPT, TEXT + " More.", path, "m", "p", max_tokens=64)  # the same first 40 tokens, so only the stamp can tell
    with pytest.raises(ValueError, match="different prompt or text"):
        Resampler(tok, PROMPT.replace("17 * 23", "seventeen times twenty-three"), TEXT, path, "m", "p", max_tokens=64)
    with open(path, "a") as f:
        f.write(json.dumps({"t": 0, "i": 5, "prompt_tokens": 20, "cost": 0.0}) + "\n")  # an older record without a stamp
    assert len(Resampler(tok, PROMPT, TEXT, path, "m", "p", max_tokens=64).rollouts) == 3 and "1 rollouts in" in capsys.readouterr().out

def test_seam_and_bounds(tok, tmp_path):
    with pytest.raises(ValueError, match="merges into the prompt"):
        Resampler(tok, PROMPT, "\n" + TEXT, str(tmp_path / "r.jsonl"), "m", "p")  # "<think>\n" + "\n" tokenizes as one "\n\n" token
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "r.jsonl"), "m", "p")
    with pytest.raises(ValueError, match="outside"):
        asyncio.run(res.rollout(55, 0))
    with pytest.raises(ValueError, match="does not retokenize"):
        asyncio.run(res.rollout(51, 0))

def test_stage_two_is_checked(tok, tmp_path, monkeypatch):
    base = fake_complete(tok, [])
    async def wrap_stage_two(prompt, model, provider, stop=None, client=None, **kw):
        b = await base(prompt, model, provider, stop=stop, client=client, **kw)
        if prompt.endswith(RESPONSE_OPEN):
            b["usage"]["prompt_tokens"] += 3
        return b
    monkeypatch.setattr(rs, "complete", wrap_stage_two)
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "r.jsonl"), "m", "p", stop="</think>", response_open=RESPONSE_OPEN)
    with pytest.raises(ExceptionGroup) as e:
        asyncio.run(res.fill({10: 1}))
    assert "on the response call" in str(e.value.exceptions[0]) and res.rollouts == []

def test_judge_cannot_overwrite_fields(tok, tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "complete", fake_complete(tok, []))
    async def judge(r):
        return {"match": True, "cost": 0.5}
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "r.jsonl"), "m", "p", judge=judge)
    with pytest.raises(ExceptionGroup) as e:
        asyncio.run(res.fill({0: 1}))
    assert isinstance(e.value.exceptions[0], ValueError) and "['cost']" in str(e.value.exceptions[0]) and os.path.exists(res.path) and not os.path.exists(res.judged_path)  # the rollout is saved, the verdict is not

def test_fill_indices_stay_unique_after_failures(tok, tmp_path, monkeypatch):
    base, calls = fake_complete(tok, []), itertools.count()
    async def flaky(*args, **kw):
        if next(calls) == 1:
            raise RequestFailed("429", "throttled")
        return await base(*args, **kw)
    monkeypatch.setattr(rs, "complete", flaky)
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "r.jsonl"), "m", "p")
    recs = asyncio.run(res.fill({0: 3}))
    assert [None if r is None else r["i"] for r in recs] == [0, None, 2]
    assert [r["i"] for r in asyncio.run(res.fill({0: 3}))] == [3]  # not 2 again
    assert sorted(r["i"] for r in res.rollouts) == [0, 2, 3]

def test_reuse(tok, tmp_path, monkeypatch):
    """A rollout that continues with the text's next k tokens is a sample from every position through t + k."""
    ks = itertools.cycle([0, 3, 44])
    async def complete(prompt, model, provider, stop=None, client=None, **kw):
        t, k = len(tok(prompt, add_special_tokens=False)["input_ids"]) - 20, next(ks)
        rec = {"finish_reason": "stop", "provider": "Fake", "model": model, "prompt_tokens": 20 + t, "completion_tokens": 5, "reasoning_tokens": None, "cost": 0.001}
        return body({"text": " zzz", "reasoning": None, **rec} if k == 0 else {"text": "**391**", "reasoning": tok.decode(res.ids[t:t + k]) + " zzz", **rec})
    monkeypatch.setattr(rs, "complete", complete)
    async def judge(r):
        return {"match": r["reasoning"] is not None}
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "r.jsonl"), "m", "p", judge=judge)
    recs = asyncio.run(res.fill({10: 3}))
    assert sorted(res.reuse(r) for r in recs) == [0, 3, 44]  # the one without reasoning is compared through its response
    sc = res.scores()
    assert [(s["t"], s["n"], s["direct"], s["k"]) for s in sc[:5]] == [(10, 3, 3, 2), (11, 2, 0, 2), (12, 2, 0, 2), (13, 2, 0, 2), (14, 1, 0, 1)] and len(sc) == 45 and sc[-1]["t"] == 54  # 51, which cannot be sampled, is covered
    assert sc[0]["ci"] == wilson(2, 3) and sc[1]["p"] == 1.0 and sc[1]["ci"] == wilson(2, 2)
    assert [(s["t"], s["n"], s["direct"]) for s in res.scores(method="naive")] == [(10, 3, 3)]
    rc = res.scores(method="recursion")  # at 10 one of three departs with False: p = 2/3 of p(11); from 11 on, every rollout on the text continues or departs with True
    assert [s["t"] for s in rc] == [s["t"] for s in sc] and rc[0]["p"] == pytest.approx(2 / 3) and all(s["p"] == 1.0 for s in rc[1:]) and rc[0]["ci"] == pytest.approx((2 / 3 - 1.96 * math.sqrt(0.148148), 2 / 3 + 1.96 * math.sqrt(0.148148)), abs=1e-5)
    with pytest.raises(ValueError, match="method"):
        res.scores(method="exact")

def test_judge_after_save(tok, tmp_path, monkeypatch):
    """A rollout is on disk before the judge sees it; a judge failure leaves it unjudged for the next pass."""
    monkeypatch.setattr(rs, "complete", fake_complete(tok, []))
    n = itertools.count()
    async def judge(r):
        if next(n) == 1:
            raise RequestFailed("429", "throttled")
        return {"match": True}
    path = str(tmp_path / "r.jsonl")
    res = Resampler(tok, PROMPT, TEXT, path, "m", "p", judge=judge)
    recs = asyncio.run(res.fill({0: 3}))
    assert [r["i"] for r in recs] == [0, 1, 2] and sum("match" in r for r in recs) == 2 and len(res.judged) == 2 and len(list(open(res.judged_path))) == 2
    with pytest.raises(RuntimeError, match="1 rollouts are not judged"):
        res.deficit(0.1)
    assert [r["match"] for r in asyncio.run(res.judge_pending())] == [True] and res.judged == {(0, 0), (0, 1), (0, 2)}
    res2 = Resampler(tok, PROMPT, TEXT, path, "m", "p", judge=judge)  # a reload merges the sidecar
    assert res2.judged == res.judged and [r["match"] for r in res2.rollouts] == [True] * 3 and asyncio.run(res2.judge_pending()) == []
    with open(tmp_path / "old.jsonl", "w") as f:  # verdicts inline, an older format: loaded as judged
        f.writelines(json.dumps(r) + "\n" for r in res2.rollouts)
    old = Resampler(tok, PROMPT, TEXT, str(tmp_path / "old.jsonl"), "m", "p", judge=judge)
    assert old.judged == res.judged and asyncio.run(old.judge_pending()) == [] and not os.path.exists(old.judged_path)
    with open(res.judged_path, "a") as f:
        f.write(json.dumps({"t": 0, "i": 9, "match": True}) + "\n")
    with pytest.raises(ValueError, match=r"rollout \(0, 9\), which"):
        Resampler(tok, PROMPT, TEXT, path, "m", "p")

def test_deficit(tok, tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "complete", fake_complete(tok, []))
    flips = itertools.cycle([True, False])
    async def judge(r):
        return {"match": next(flips)}
    res = Resampler(tok, PROMPT, TEXT, str(tmp_path / "r.jsonl"), "m", "p", judge=judge)
    assert res.deficit(0.1) == {t: 5 for t in range(55) if t != 51} and res.deficit(0.1, n_min=20, batch=8, positions=[3, 7]) == {3: 8, 7: 8}
    asyncio.run(res.fill({0: 100, 20: 4}))
    d = res.deficit(0.1)  # p = 0.5 at t=0 with 100 samples: half-width 0.098
    assert 0 not in d and d[20] == 9 and d[1] == 5
    assert res.deficit(0.05)[0] == 105 and 0 not in res.deficit(0.05, n_max=100)
    assert res.deficit(0.5, n_min=20)[20] == 9 and 0 not in res.deficit(0.5, n_min=20)  # the floor is asked for a batch at a time
    assert res.deficit(0.5, n_min=6) == {t: 5 for t in range(55) if t not in (0, 20, 51)} | {20: 6}

def test_estimate():
    """A hand-checked pool on a 3-token text: (t, k, outcome) rollouts sampled from t that continue with the text's next k tokens."""
    rollouts = [(0, 3, True), (0, 0, False), (1, 1, True), (2, 0, None), (3, 0, False)]
    counts = lambda est: [(s["t"], s["n"], s["direct"], s["k"], s["judged"], s["other"]) for s in est]
    naive = estimate(3, rollouts, "naive")
    assert counts(naive) == [(0, 2, 2, 1, 2, 0), (1, 1, 1, 1, 1, 0), (2, 1, 1, 0, 0, 1), (3, 1, 1, 0, 1, 0)] and [s["p"] for s in naive][:2] == [0.5, 1.0] and math.isnan(naive[2]["p"]) and naive[3]["p"] == 0.0 and naive[0]["ci"] == wilson(1, 2)
    reuse = estimate(3, rollouts)
    assert counts(reuse) == [(0, 2, 2, 1, 2, 0), (1, 2, 1, 2, 2, 0), (2, 3, 1, 2, 2, 1), (3, 2, 1, 1, 2, 0)] and [s["p"] for s in reuse] == [0.5, 1.0, 1.0, 0.5]
    rc = estimate(3, rollouts, "recursion")
    assert counts(rc) == counts(reuse)
    # from the end: p(3) = 1/2 over the two that get there, variance (1/2)(1/2)/2. At 2 one of the two judged departs with True and one continues: p = 1/2 + p(3)/2; at 1 both continue: p = p(2); at 0 one of two departs with False: p = p(1)/2
    assert [s["p"] for s in rc] == pytest.approx([0.375, 0.75, 0.75, 0.5])
    v3 = 0.125
    v2 = 0.5 * (2 / 3) * (1 / 3) / 2 + 0.25 * v3 + 0.5 * 0.5 * (0.5 - 1) ** 2 / 2
    v0 = 0.5 * (1 / 3) * (2 / 3) / 2 + 0.25 * v2 + 0.5 * 0.5 * (0.75 - 0) ** 2 / 2
    assert [(s["ci"][1] - s["ci"][0]) / 3.92 for s in rc] == pytest.approx([math.sqrt(v0), math.sqrt(v2), math.sqrt(v2), math.sqrt(v3)])
    with pytest.raises(ValueError, match="method"):
        estimate(3, rollouts, "exact")

def test_estimate_without_reuse_is_naive():
    """When no rollout continues with the text's next token, every position's rollouts depart there and the recursion is the plain rate, with its shrunk-variance interval."""
    rollouts = [(0, 0, True), (0, 0, True), (2, 0, False), (2, 0, True), (3, 0, None)]
    rc, naive = estimate(3, rollouts, "recursion"), estimate(3, rollouts, "naive")
    assert [s["p"] for s in rc][:2] == [s["p"] for s in naive][:2] == [1.0, 0.5] and math.isnan(rc[2]["p"]) and math.isnan(naive[2]["p"])
    assert rc[0]["ci"] == pytest.approx((1 - 1.96 * math.sqrt(0.75 * 0.25 / 2), 1 + 1.96 * math.sqrt(0.75 * 0.25 / 2)))

def test_recursion_is_unbiased_with_a_calibrated_interval():
    """Rollouts simulated from a known process (per position, the chance of continuing with the text's token and the rate of True on departing), three per position on a 30-token text, 300 trials: the recursion's error against the truth averages zero, its z (error over the interval's sd) has mean square about 1 and 95% within 1.96, and its squared error is below reuse's, which is below naive's."""
    rng = np.random.default_rng(0)
    T, S, trials = 30, 3, 300
    q, r = rng.uniform(0.6, 1.0, T), rng.uniform(0, 1, T + 1)  # r[T]: the rate among rollouts that reproduce the whole text
    p = np.empty(T + 1)
    p[T] = r[T]
    for s in range(T - 1, -1, -1):
        p[s] = q[s] * p[s + 1] + (1 - q[s]) * r[s]
    err, z = {m: [] for m in ("naive", "reuse", "recursion")}, []
    for _ in range(trials):
        rolls = []
        for t0 in range(T + 1):
            for _ in range(S):
                s = t0
                while s < T and rng.random() < q[s]:
                    s += 1
                rolls.append((t0, s - t0, bool(rng.random() < r[s])))
        for m in err:
            est = estimate(T, rolls, m)
            err[m] += [e["p"] - p[e["t"]] for e in est]
            if m == "recursion":
                z += [(e["p"] - p[e["t"]]) / ((e["ci"][1] - e["ci"][0]) / 3.92) for e in est]
    mse = {m: np.mean(np.square(err[m])) for m in err}
    z = np.array(z)
    assert mse["recursion"] < 0.8 * mse["reuse"] < 0.8 * mse["naive"], mse
    assert abs(np.mean(err["recursion"])) < 0.003 and abs(z.mean()) < 0.05, (np.mean(err["recursion"]), z.mean())
    assert 0.85 < np.mean(z ** 2) < 1.2 and 0.92 < np.mean(np.abs(z) < 1.96) < 0.98, (np.mean(z ** 2), np.mean(np.abs(z) < 1.96))

K_TOKENIZERS = ["Qwen/Qwen3-0.6B", "meta-llama/Llama-3.2-1B-Instruct", "google/gemma-3-1b-it"]  # two byte-level BPEs and a sentencepiece
K_PROMPT, K_TEXT = "Question: what is 17 * 23?\nAnswer:\n", "Okay, 17 * 23: 17 * 20 = 340,   17 * 3 = 51,\n\n\tso 340 + 51 = 391. 𐍈 done, 1234567 and  890 unbelievably."

@pytest.mark.hf
@pytest.mark.parametrize("name", K_TOKENIZERS)
def test_k_ids_and_k_joint(name):
    """The reuse count from sampled ids (k_ids, right by construction) against the one from text (k_joint), on cases built from ids so the truth is known. They agree wherever the text tokenizes back to the sampled ids; k_joint is lower for ids that end inside a multi-byte character and higher for a sampled split the tokenizer would not produce."""
    tk = load_tokenizer(name)
    enc = lambda s: tk(s, add_special_tokens=False)["input_ids"]
    p_ids = enc(K_PROMPT)
    ids = enc(K_PROMPT + K_TEXT)[len(p_ids):]
    trace, T = {"prompt": K_PROMPT, "prompt_ids": p_ids, "ids": ids}, len(ids)
    pre = [tk.decode(ids[:t]) for t in range(T + 1)]
    clean = [t for t in range(T + 1) if enc(K_PROMPT + pre[t]) == p_ids + ids[:t]]  # the cuts a Resampler samples from
    text = lambda t, c: tk.decode(ids[:t] + c)[len(pre[t]):]  # a continuation as a provider returns it
    assert len(clean) > 0.9 * T and all(k_ids(ids[t:], ids, t) == k_joint(tk, trace, t, text(t, ids[t:])) == T - t for t in clean)  # the rest of the trace, through its runs of spaces and digits
    dep, stable, cases = enc(" zebra"), 0, 0
    for t in clean:
        for k in (0, 1, 3, 6):
            if t + k < T:
                c = ids[t:t + k] + dep  # the trace's next k tokens, then a departure
                assert k_ids(c, ids, t) == k
                cases += 1
                if joint_ids(tk, trace, [(t, text(t, c))]) == [c]:
                    stable += 1
                    assert k_joint(tk, trace, t, text(t, c)) == k
    assert stable > 0.7 * cases
    assert k_ids([], ids, 3) == k_joint(tk, trace, clean[2], "") == k_ids([], ids, T) == k_joint(tk, trace, T, "") == 0 and joint_ids(tk, trace, []) == []  # an empty continuation, and the end of the trace
    assert joint_ids(tk, trace, [(1, "s")]) == [None] and k_joint(tk, trace, 1, "s") == 0  # "Okay" + "s": the continuation merges into the prefix's last token
    inside = next(j for j in range(T + 1) if "�" in pre[j])  # a cut inside the multi-byte character
    t0 = max(t for t in clean if t < inside)
    assert k_ids(ids[t0:inside], ids, t0) == inside - t0 > k_joint(tk, trace, t0, text(t0, ids[t0:inside]))  # the ids match through the cut; their text ends in a replacement character
    odd = next(a + b for j in range(1, 4) for a, b in [(enc("Okay"[:j]), enc("Okay"[j:]))] if len(a) == len(b) == 1)  # "Okay" sampled as two tokens, where the tokenizer makes one
    assert k_ids(odd + ids[1:3], ids, 0) == 0 < k_joint(tk, trace, 0, text(0, odd + ids[1:3]))  # the text is the trace's, so k_joint counts tokens the rollout did not sample

def trace_of(tok, cap=None) -> dict:
    p_ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    return {"prompt": PROMPT, "prompt_ids": p_ids, "ids": tok(PROMPT + TEXT, add_special_tokens=False)["input_ids"][len(p_ids):], "text": TEXT, "cap": cap}

def api_record(t: int, i: int, response: str, stopped: bool = True, reasoning: str | None = None, **verdict) -> dict:
    return {"t": t, "i": i, "reasoning": reasoning, "response": response, "finish_reason": "stop" if stopped else "length", "provider": "Fake", "prompt_tokens": 20 + t, "completion_tokens": 7, "cost": 0.001, "cfg": {}, "raw": [], **verdict}

def write_jsonl(path, recs: list[dict]) -> str:
    open(path, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    return str(path)

def test_load_rollouts_reads_both_shapes_alike(tok, tmp_path):
    """The same rollouts stored as local records and as Resampler records load to the same ids, text, k, stopped and capped, with one cap rule for both."""
    trace = trace_of(tok, cap=30)
    ids, zzz = trace["ids"], tok(" zzz", add_special_tokens=False)["input_ids"]
    conts = [(20, 0, ids[20:29], True), (20, 1, ids[20:30], True), (20, 2, ids[20:31], True), (20, 3, ids[20:25] + zzz, False), (10, 0, zzz, True)]  # t + len(ids) one below the cap, at it and above it; one that did not stop; one that leaves the trace at once
    local = load_rollouts(trace, write_jsonl(tmp_path / "x_local.jsonl", [{"t": t, "i": i, "ids": c, "ended": s} for t, i, c, s in conts]), tok)
    api = load_rollouts(trace, write_jsonl(tmp_path / "x_api.jsonl", [api_record(t, i, tok.decode(c), s) for t, i, c, s in conts]), tok)
    same = lambda rows: [(r["t"], r["i"], r["ids"], r["text"], r["k"], r["stopped"], r["capped"]) for r in rows]
    assert same(local) == same(api) == [(t, i, c, tok.decode(c), k, s, cap) for (t, i, c, s), k, cap in zip(conts, [9, 10, 11, 5, 0], [False, True, True, True, False])]
    assert [r["tokens"] for r in local] == [10, 11, 12, 5 + len(zzz), len(zzz) + 1] and [r["tokens"] for r in api] == [7] * 5  # the ids plus one for a stop; the provider's count
    assert local[0]["raw"] == {"t": 20, "i": 0, "ids": ids[20:29], "ended": True} and api[0]["raw"]["response"] == tok.decode(ids[20:29])
    assert [r["capped"] for r in load_rollouts(trace | {"cap": None}, str(tmp_path / "x_api.jsonl"), tok)] == [False, False, False, True, False]  # no cap: only the rollout that did not stop

def test_load_rollouts_text_of_a_resampler_record(tok, tmp_path):
    """The continuation of a Resampler record is its reasoning where it holds one, else its response; one that merges into the prefix has k = 0 and its ids from the text alone."""
    trace = trace_of(tok)
    rows = load_rollouts(trace, write_jsonl(tmp_path / "r.jsonl", [api_record(10, 0, "**391**", reasoning=tok.decode(trace["ids"][10:12]) + " zzz"), api_record(10, 1, " zzz"), api_record(1, 0, "s")]), tok)
    assert [(r["text"], r["k"]) for r in rows] == [(tok.decode(trace["ids"][10:12]) + " zzz", 2), (" zzz", 0), ("s", 0)]
    assert rows[2]["ids"] == tok("s", add_special_tokens=False)["input_ids"] and joint_ids(tok, trace, [(1, "s")]) == [None]  # "Okay" + "s" is another token

def test_load_rollouts_verdicts_and_failures(tok, tmp_path):
    """Inline verdicts and the sidecar's load the same; a verdict without its rollout, a repeated (t, i), a record of neither shape and a verdict named like a reader field raise."""
    trace, zzz = trace_of(tok), tok(" zzz", add_special_tokens=False)["input_ids"]
    recs = [{"t": 3, "i": 0, "ids": zzz, "ended": True}, {"t": 3, "i": 1, "ids": zzz, "ended": False}]
    verdicts = [{"match": True, "answer": 391}, {"match": None, "answer": None}]
    inline = load_rollouts(trace, write_jsonl(tmp_path / "a.jsonl", [r | v for r, v in zip(recs, verdicts)]), tok)
    path = write_jsonl(tmp_path / "b.jsonl", recs)
    assert all("match" not in r for r in load_rollouts(trace, path, tok))
    write_jsonl(tmp_path / "b.judged.jsonl", [{"t": r["t"], "i": r["i"]} | v for r, v in zip(recs, verdicts)])
    side = load_rollouts(trace, path, tok)
    drop_raw = lambda rows: [{f: v for f, v in r.items() if f != "raw"} for r in rows]
    assert drop_raw(inline) == drop_raw(side) and [(r["match"], r["answer"]) for r in side] == [(True, 391), (None, None)] and side[0]["raw"] == recs[0]
    assert [s["p"] for s in rollout_scores(trace, side, tok)] == [1.0] and rollout_scores(trace, side, tok, lambda r: r["stopped"])[0]["k"] == 1  # a verdict field, or a function of the rollout
    write_jsonl(tmp_path / "b.judged.jsonl", [{"t": 9, "i": 0, "match": True}])
    with pytest.raises(ValueError, match="does not hold"):
        load_rollouts(trace, path, tok)
    for bad, why in (([recs[0], recs[0]], "repeat"), ([{"t": 3, "i": 0, "text": "x"}], "neither ids nor response"), ([recs[0] | {"k": 2}], "named like")):
        with pytest.raises(ValueError, match=why):
            load_rollouts(trace, write_jsonl(tmp_path / "c.jsonl", bad), tok)

def test_rollout_scores_equal_resampler_scores(tok, tmp_path, monkeypatch):
    """The reader over a Resampler's file gives that Resampler's reuse counts and, through rollout_scores, its scores for all three methods."""
    ks = itertools.cycle([0, 3, 44, 7])
    async def complete(prompt, model, provider, stop=None, client=None, **kw):
        t, k = len(tok(prompt, add_special_tokens=False)["input_ids"]) - 20, next(ks)
        rec = {"finish_reason": "stop", "provider": "Fake", "model": model, "prompt_tokens": 20 + t, "completion_tokens": 5, "reasoning_tokens": None, "cost": 0.001}
        return body({"text": " zzz", "reasoning": None, **rec} if k == 0 else {"text": "**391**", "reasoning": tok.decode(res.ids[t:t + k]) + " zzz", **rec})
    monkeypatch.setattr(rs, "complete", complete)
    async def judge(r):
        return {"match": r["reasoning"] is not None and r["i"] != 2}
    path = str(tmp_path / "r.jsonl")
    res = Resampler(tok, PROMPT, TEXT, path, "m", "p", judge=judge)
    asyncio.run(res.fill({0: 4, 10: 4, 30: 3}))
    trace = {"prompt": PROMPT, "prompt_ids": res.prompt_ids, "ids": res.ids, "text": TEXT, "cap": None}
    rollouts = load_rollouts(trace, path, tok)
    assert [r["k"] for r in rollouts] == [res.reuse(r) for r in res.rollouts] and sorted({r["k"] for r in rollouts}) == [0, 3, 7, 24, 44]
    for method in ("naive", "reuse", "recursion"):
        assert rollout_scores(trace, rollouts, tok, "match", method) == res.scores("match", method) == rollout_scores(trace, rollouts, tok, lambda r: r["match"], method)

class Sharp:
    """The tiny model with its logits scaled until its next-token distributions have shape (a top token near 0.2, a third of the mass on tokens under 1%), and two stop tokens at about 3% each per step."""
    EOS = [5, 9]

    def __init__(self, model):
        self.model, self.generation_config = model, SimpleNamespace(eos_token_id=self.EOS)

    def __call__(self, toks, return_type="logits", **kw):
        out = self.model(toks, return_type=return_type, **kw)
        logits = (out[0] if return_type == "logits_and_cache" else out) * 60
        logits[..., self.EOS] = logits.logsumexp(-1, keepdim=True) - 3.5
        return (logits, out[1]) if return_type == "logits_and_cache" else logits

    def __getattr__(self, name):
        return getattr(self.model, name)

CALIB_TRACE = {"prompt_ids": [1, 100, 200, 300], "ids": [400, 500, 600, 700, 800, 900, 1000, 1100]}

def test_calib_rollouts_equals_a_direct_computation(tiny_bridge):
    """Each rollout's scores against the formulas applied to one uncached forward of its own sequence in float64: a stopped rollout (its stop scored, the eos ids merged), one past the 100-step chunk that hit the cap (no stop scored), and an empty one that stopped at once. The scores do not depend on what a row is batched with."""
    model = Sharp(tiny_bridge)
    eos = sorted(eos_ids(model))
    assert len(eos) == 3
    rollouts = [{"t": 2, "i": 0, "ids": [11, 12, 13], "stopped": True, "raw": {}}, {"t": 6, "i": 0, "ids": t.randint(10, 30000, (130,), generator=t.Generator().manual_seed(0)).tolist(), "stopped": False, "raw": {}}, {"t": 0, "i": 1, "ids": [], "stopped": True, "raw": {}}]
    def direct(r):
        x = r["ids"] + eos[-1:] * r["stopped"]
        lp = model(t.tensor([CALIB_TRACE["prompt_ids"] + CALIB_TRACE["ids"][:r["t"]] + r["ids"]]))[0, 3 + r["t"]:3 + r["t"] + len(x)].double().log_softmax(-1)
        stop = lp[:, eos].logsumexp(-1)
        lp[:, eos] = -math.inf
        lp[:, eos[-1]] = stop
        p = lp.exp()
        plp, lx = t.where(p > 0, p * lp, 0.0), lp[t.arange(len(x)), x]
        H, a, ps = -plp.sum(-1), (p * (p < 0.01)).sum(-1), p[:, eos[-1]]
        return {"t": r["t"], "i": r["i"], "n": len(x), "s": (lx + H).sum().item(), "v": ((plp * t.where(p > 0, lp, 0.0)).sum(-1) - H * H).sum().item(), "rare": [float((lx < math.log(0.01)).sum()), a.sum().item(), (a * (1 - a)).sum().item()], "stop": [float(r["stopped"]), ps.sum().item(), (ps * (1 - ps)).sum().item()]}
    want = {(r["t"], r["i"]): direct(r) for r in rollouts}
    assert [w["n"] for w in want.values()] == [4, 130, 1] and want[6, 0]["rare"][0] > 100 and 0.05 < want[2, 0]["stop"][1] < 0.5
    for batch_size in (1, 3):
        got = calib_rollouts(model, CALIB_TRACE, rollouts, batch_size, quiet=True)
        assert [(c["t"], c["i"]) for c in got] == [(0, 1), (2, 0), (6, 0)]  # shortest sequence first
        for c in got:
            w = want[c["t"], c["i"]]
            assert c["n"] == w["n"] and c["rare"][0] == w["rare"][0] and c["stop"][0] == w["stop"][0]
            assert [c["s"], c["v"], *c["rare"][1:], *c["stop"][1:]] == pytest.approx([w["s"], w["v"], *w["rare"][1:], *w["stop"][1:]], rel=1e-4, abs=1e-4)

def test_calib_check_passes_the_models_own_samples_and_flags_altered_ones(tiny_bridge):
    """Rollouts sampled from the model by stream_rolling, and by a plain uncached loop, pass. A higher temperature, sampling from the top 5 tokens, and rollouts ended where the model would not are each flagged by the score meant for them."""
    model = Sharp(tiny_bridge)
    eos = eos_ids(model)
    prefix = lambda cut: CALIB_TRACE["prompt_ids"] + CALIB_TRACE["ids"][:cut]
    def sample(transform):
        out = []
        for cut in (0, 4, 8):
            seqs = t.tensor([prefix(cut)] * 40)
            for _ in range(24):
                seqs = t.cat([seqs, t.multinomial(transform(model(seqs)[:, -1]).softmax(-1), 1)], 1)
            for i, row in enumerate(seqs[:, 4 + cut:].tolist()):
                end = next((j for j, x in enumerate(row) if x in eos), None)
                out.append({"t": cut, "i": i, "ids": row[:end], "stopped": end is not None, "raw": {}})
        return out
    check = lambda rollouts: calib_check(calib_rollouts(model, CALIB_TRACE, rollouts, 16, quiet=True))
    t.manual_seed(0)
    own = [{"t": cut, "i": i, "ids": ids, "stopped": len(ids) < 24, "raw": {}} for cut in (0, 4, 8) for i, ids in enumerate(sample_rolling(model, t.tensor([prefix(cut)]), 40, 40, 24, quiet=True))]
    c = check(own)
    assert c["rollouts"] == 120 and c["tokens"] > 1000 and not c["flagged"] and abs(c["temperature"] - 1) < 0.05 and 0.3 < sum(r["stopped"] for r in own) / 120 < 0.95
    assert not check(sample(lambda lg: lg))["flagged"]
    hot = check(sample(lambda lg: lg / 1.3))
    assert hot["flagged"] and hot["z"]["tokens"] < -5 and hot["z"]["rare"] > 3 and hot["temperature"] > 1.15
    top = check(sample(lambda lg: lg.masked_fill(lg < lg.topk(5).values[:, -1:], -math.inf)))
    assert top["flagged"] and top["z"]["rare"] < -5 and top["z"]["tokens"] > 5 and top["temperature"] < 0.9
    early = check([r | {"ids": r["ids"][:2], "stopped": True} for r in own])
    assert early["flagged"] and early["z"]["stops"] > 5

def test_calib_check_sums_and_edges():
    a = [{"t": 0, "i": 0, "n": 10, "s": -2.0, "v": 4.0, "rare": [1.0, 0.5, 0.4], "stop": [1.0, 0.7, 0.2]}]
    b = [{"t": 0, "i": 1, "n": 30, "s": 1.0, "v": 5.0, "rare": [0.0, 0.5, 0.6], "stop": [0.0, 0.2, 0.05]}]
    c = calib_check(a + b)
    assert c == {"rollouts": 2, "tokens": 40, "mean_s": pytest.approx(-1 / 40), "temperature": pytest.approx(1 + -1 / (2 * -1 - 9)), "z": {"tokens": pytest.approx(-1 / 3), "rare": pytest.approx(0.0), "stops": pytest.approx(0.1 / 0.5)}, "flagged": False}
    assert calib_check(a)["z"]["tokens"] == -1.0 and calib_check(a, z_max=0.9)["flagged"] and not calib_check(a, z_max=1.2)["flagged"]
    still = [a[0] | {"rare": [3.0, 3.0, 0.0], "stop": [1.0, 1.0, 0.0]}]  # nothing uncertain about rare tokens or stops, and what was seen is what was expected
    assert calib_check(still)["z"] == {"tokens": -1.0, "rare": 0.0, "stops": 0.0}
    off = calib_check([a[0] | {"stop": [1.0, 0.0, 0.0]}])  # a stop where the model gives one no chance
    assert off["z"]["stops"] == math.inf and off["flagged"]
    with pytest.raises(ValueError, match="reasoning"):
        calib_rollouts(None, {}, [{"t": 0, "i": 0, "ids": [1], "stopped": True, "raw": {"reasoning": "thinking"}}])

def test_resample_curve_overlays_and_plain_estimates():
    sc = estimate(6, [(0, 6, True), (0, 2, True), (0, 0, False), (3, 3, True), (3, 1, False), (5, 1, True)], "recursion")  # estimate's own output: no token
    assert all("token" not in s for s in sc) and any(s["ci"][1] > 1 for s in sc)
    fig = resample_curve(sc, return_fig=True)
    assert len(fig.data) == 2 and fig.layout.paper_bgcolor == DARK["paper_bgcolor"] and all(0 <= y <= 1 for y in fig.data[0].y)  # the band is clipped
    assert fig.data[1].text[0].startswith("t=0<br>p=") and not fig.data[1].showlegend
    named = {f"source {j}": sc for j in range(5)}
    fig = resample_curve(named, return_fig=True)
    assert len(fig.data) == 10 and [d.name for d in fig.data[1::2]] == list(named) and [d.line.dash for d in fig.data[1::2]] == ["solid"] * 4 + ["dash"] and fig.data[9].line.color == fig.data[1].line.color == SERIES[0] and fig.data[9].text[0].startswith("source 4: t=0<br>")
    with pytest.raises(ValueError, match="9 curves"):
        resample_curve({str(j): sc for j in range(9)}, return_fig=True)
