"""Token-level resampling through OpenRouter's raw /completions: cut a CoT or a response at token positions, sample continuations per prefix into a rollouts jsonl that reruns top up per position, judge them, and score P(outcome | prefix_t) with Wilson intervals.

The model continues the CoT only if the provider feeds it exactly the rendered prompt string, and nothing in a response says whether it did. Before a paid run, in this order:

1. Render the prompt so it ends inside the open think block (or the open text block, to cut a reasoning-off response). Qwen: apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True) ends "<|im_start|>assistant\\n<think>\\n". DeepSeek V4 ships encoding/encoding_dsv4.py instead of a template (hf_hub_download it, sys.path.insert its directory); encode_messages(msgs, "thinking") ends "<｜Assistant｜><think>". Inkling's template ends at <|message_model|>, so append <|content_thinking|> (or <|content_text|> for a response cut).
   Chat-message prefill, a trailing assistant message holding the partial CoT, never continues it: every template closes the block, the model starts over or answers, and some providers render EOS so the model emits garbage. Count tokens with add_special_tokens=False and know whether the tokenizer adds BOS on its own, since the provider's count is compared with the local one.
2. probe(tokenizer, model, render) on every endpoint, a few cents. Pass = prompt_tokens exactly equal to the local count on every string, and every continuation of "... 340 + 51 =" starting with 391. Failures: an offset of a few tokens, constant give or take one from tokens merging at the wrapper's boundaries, is a chat wrapper (the model answers from scratch: "The product of 17 and 23 is 391."); about +80 is an injected system prompt on top; offsets that vary widely mean a different tokenizer; a count that varies between identical requests is a heterogeneous backend (probe sends each string once, so repeat it to see this); a 4xx means the endpoint is not served raw, and exhausted retries mean it was throttled or down at probe time. Exact counts with restarts ("Here's a thinking process:", "The user wants...") mean the model did not see an open think block: the local rendering is wrong, and nothing at run time catches that.
   Continuations arriving in text rather than reasoning mean the provider returns reasoning and response merged with the special tokens stripped (Together): set stop to the end-of-reasoning token and response_open to the string that opens the response block, and every rollout is two calls. Special tokens in returned text mean this provider does not strip them. Rosters change: on 2026-09-19 Qwen3.6-27B passed on Chutes and Phala and DeepSeek V4 Flash on Parasail and Mancer, while CoreWeave, which passed for both on 2026-09-02, no longer served either.
3. Resample on the endpoint that produced the base rollouts, or accept that the curve measures the resampling endpoint's deployment: quantization (fp8, fp4, unlisted) and sampling defaults differ. Pass top_p=1.0 and top_k=0 explicitly: CoreWeave applied Qwen's generation_config (top_k 20, top_p 0.95) when they were omitted, Chutes and Phala did not; sampling_defaults measures this. seed is honored by some providers and ignored by others. Cross-provider curves agreed within Wilson intervals at S=50; base-rate differences below that resolution went undetected.
4. max_tokens must cover reasoning plus response: Inkling traces exhausted 8192 and produced empty responses that looked like throttling. Such rollouts finish with "length" and are not retried; they count as other in scores only if the judge returns None (or omits the key) for them, and a judge that returns False for an empty response biases p downward. A few-hundred-token trace at S=50 stride 1 costs tens of dollars on a $2/M model, and a judge costs about as much as the subject on cheap ones.
5. At run time every rollout checks prompt_tokens == n_prompt + t (and the response call's count on a two-stage provider) and raises otherwise, catching wrapping, re-tokenization, or a provider change mid-run; inside fill that raise stops the batch (as an ExceptionGroup), with the rollouts finished so far on disk. finish_reason "error" (a mid-stream abort whose usage is wrong too) is retried inside complete. Resampler does not check the base record; in the project, check it against the same tokenizer: len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) == its prompt_tokens (plus a known constant, e.g. Inkling's appended block), and its completion_tokens minus the local tokens of reasoning + response equal to the provider's constant (1 or 2 for DeepSeek and Qwen: closing tag and EOS; 5 to 7 for Inkling), a larger gap being a truncated trace. usage reasoning_tokens is 0 or wrong on many providers: count from text.
   Cuts inside a multi-byte character (byte-level BPE) do not retokenize and are skipped. Providers throttle in bursts (DeepInfra 429s independent of request rate, Together 503s above ~12 concurrent two-stage rollouts): keep concurrency at 12 to 32 and call fill again. Prompt logprobs and echo are unavailable on the raw endpoint, so prompt identity rests on counts plus the continuation check. Closed-lab models return summarized or encrypted reasoning and cannot be resampled.
6. Before the rollouts are analyzed on local weights or pooled with local rollouts, check that the provider samples as the local model does: load_rollouts on about 150 of them, calib_rollouts with the local model, calib_check. A provider can pass every check above and still sample from another distribution: on 2026-10-03 DeepInfra's Llama-3.1-8B put P(correct) 0.23 below the local model's and Parasail's Llama-3.2-3B ended responses early, while all seven Qwen3.5-9B and Qwen3.8-27B deployments tested matched. calib_check's docstring says what a pass covers."""

import asyncio
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable

import httpx
import plotly.graph_objects as go
import torch as t

from mechtools.bars import pbar
from mechtools.colors import *
from mechtools.openrouter import RequestFailed, _usd, complete, endpoints, flat, gather_bar
from mechtools.plots import DARK, SERIES
from mechtools.sampling import eos_ids
from mechtools.stats import wilson
from mechtools.tables import show_table

ROLLOUT_KEYS = {"t", "i", "reasoning", "response", "finish_reason", "provider", "prompt_tokens", "completion_tokens", "cost", "cfg", "raw"}  # a rollout record's own fields; anything else on one is a verdict

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def estimate(T: int, rollouts: list[tuple[int, int, bool | None]], method: str = "reuse") -> list[dict]:
    """P(outcome | the first t tokens of a T-token text) at every position with samples, sorted by t, from rollouts given as (t, k, outcome): sampled from t, continuing with the text's next k tokens before departing from it (k = T - t when it reproduces the rest), and the outcome True, False, or None for no verdict. Per position: t, n samples, direct (the rollouts sampled at t itself), k with outcome True, judged (True or False), other (the rest), p and ci. method:
      naive      the rollouts sampled at t; p = k / judged, Wilson ci.
      reuse      a rollout counts at every position through t + k: conditional on producing the text's next k tokens, the rest of it is drawn from the model at t + k, exactly. n is then the effective count, and positions never sampled directly appear; p = k / judged, Wilson ci.
      recursion  the reuse counts read backward from the end: p(t) = q p(t+1) + (1 - q) r, with q the share of the judged rollouts on the text at t that continue with its next token and r the rate of True among those that depart there, so what the rollouts sampled after t say about p(t+1) reaches p(t); p(T) is the rate over everything that gets there. The ci is p +- 1.96 sd, where the variance counts the departures' r(1 - r) with r shrunk toward 1/2 (as (hits + 1) / (departures + 2)), so one departing rollout does not read as certainty, the spread between p(t+1) and r, and q^2 times the variance of p(t+1). The interval is not clipped to [0, 1]. Rollouts without a verdict are left out of this estimate, where the other two count them as other.
    Positions no rollout reaches are absent from the output; the recursion runs through such a gap unchanged, since every rollout before it departs before it. Reuse and the recursion are exact only when a rollout's distribution does not depend on where it was cut, so a length cap must rarely bind (Resampler counts max_tokens from the cut; sampling.stream_rollouts caps the total length). Measured on math traces, the recursion needs about half the tokens of reuse at equal error and the reuse about a quarter of naive at stride 1."""
    if method not in ("naive", "reuse", "recursion"):
        raise ValueError(f"method must be naive, reuse or recursion, not {method!r}")
    vals, direct, leave = defaultdict(list), Counter(), defaultdict(list)  # per position: the outcome of every rollout on the text there; the rollouts sampled there; the outcome of every rollout that departs there (at T, everything that gets there)
    for t, k, y in rollouts:
        if method == "naive":
            k = 0
        direct[t] += 1
        for s in range(t, t + k + 1):
            vals[s].append(y)
        leave[t + k].append(y)
    out = {t: {"t": t, "n": len(v), "direct": direct[t], "k": v.count(True), "judged": v.count(True) + v.count(False), "other": len(v) - v.count(True) - v.count(False)} for t, v in vals.items()}
    for s in out.values():
        s["p"], s["ci"] = s["k"] / s["judged"] if s["judged"] else float("nan"), wilson(s["k"], s["judged"])
    if method == "recursion":
        p, v = [float("nan")] * (T + 2), [float("nan")] * (T + 2)
        for t in range(T, -1, -1):
            if t not in out or not out[t]["judged"]:
                continue
            n, dep, hit = out[t]["judged"], leave[t].count(True) + leave[t].count(False), leave[t].count(True)
            m, r, rs = dep / n, hit / max(dep, 1), (hit + 1) / (dep + 2)  # the share departing at t, their rate of True, and that rate shrunk for the variance
            p[t], v[t] = m * r, m * rs * (1 - rs) / n
            if m < 1:  # some continue to t + 1, so that estimate carries them, and the gap between staying and departing adds variance
                p[t] += (1 - m) * p[t + 1]
                v[t] += (1 - m) ** 2 * v[t + 1] + m * (1 - m) * (p[t + 1] - r) ** 2 / n
            out[t]["p"], out[t]["ci"] = p[t], (p[t] - 1.96 * math.sqrt(v[t]), p[t] + 1.96 * math.sqrt(v[t]))
    return [out[t] for t in sorted(out)]

def k_ids(ids: list[int], trace_ids: list[int], t: int) -> int:
    """The reuse count of a rollout from t whose sampled ids are known: how many of the trace's tokens after its first t the continuation starts with."""
    k = 0
    while k < len(ids) and t + k < len(trace_ids) and ids[k] == trace_ids[t + k]:
        k += 1
    return k

def joint_ids(tokenizer, trace: dict, cuts: list[tuple[int, str]]) -> list[list[int] | None]:
    """A continuation's ids recovered from its text, per (t, continuation): prompt + the trace's first t tokens + the continuation tokenized as one string, less the prompt's and the prefix's ids. None where that string no longer starts with those ids, the continuation having merged into the prefix's last token. trace needs prompt, prompt_ids and ids.
    These are the tokenizer's split of the text, which is not always the split the model sampled: measured on local rollouts, 2 to 4% of Llama-3 rollouts and 0.1% of Qwen3.5 rollouts hold a token pair the tokenizer would split otherwise, nearly always after the rollout has left the trace (the reuse count from these ids was wrong in 1 of 8,600)."""
    n, pre = len(trace["prompt_ids"]), {t: tokenizer.decode(trace["ids"][:t]) for t in {t for t, _ in cuts}}
    full = tokenizer([trace["prompt"] + pre[t] + c for t, c in cuts], add_special_tokens=False)["input_ids"] if cuts else []
    return [ids[n + t:] if ids[:n + t] == trace["prompt_ids"] + trace["ids"][:t] else None for (t, _), ids in zip(cuts, full)]

def k_joint(tokenizer, trace: dict, t: int, continuation: str) -> int:
    """The reuse count of a rollout known only as text: k_ids of its joint_ids, and 0 where the continuation merged into the prefix, so it did not produce the trace's next token."""
    ids = joint_ids(tokenizer, trace, [(t, continuation)])[0]
    return 0 if ids is None else k_ids(ids, trace["ids"], t)

class Resampler:
    """Resamples text (a CoT, or a reasoning-off response) from token position t: the provider continues prompt + the first t tokens of text through raw /completions, so prompt must be the rendered chat template ending inside the open think or text block, and provider must pass raw prompts through verbatim: run probe first, and read the module docstring for the rest of the checklist. text is tokenized after the prompt, as the model produced it, and must not merge into the prompt's last token. Each rollout checks the provider's prompt_tokens against the local count of the exact string sent (on the response call too) and raises otherwise.
    Rollouts append to the jsonl at path, one per line: t, i, reasoning, response, finish_reason, provider, prompt_tokens, completion_tokens, cost, cfg (the configuration stamp: model, provider, stop, response_open, kw, hashes of prompt and text), raw (the response bodies, one per call). Verdicts append to the sidecar next to it (path with .judged.jsonl in place of .jsonl), one line per judged rollout: t, i, plus whatever judge(rollout) returns, which may not reuse the rollout's names; loading merges them into the records, so a rollout in memory carries its verdict's fields. judge is async (awaited) and receives only the record; response is the continuation only, so when text is a response, a judge that needs the whole response rebuilds it as resampler.prefix(rollout["t"]) + rollout["response"]. The judge is called on every rollout, including empty and truncated ones, and decides what to return for them (None or a missing key counts as other in scores).
    A rollout is on disk before it is judged, so a judge outage costs nothing: fill judges every unjudged rollout after sampling, and judge_pending does the same on its own. A judge that raises RequestFailed leaves its rollout unjudged for the next pass; any other exception from the judge stops the pass. A file whose records carry verdict fields inline (an older format) loads as judged.
    Loading a file checks every record's token counts and stamp against the instance and raises on a mismatch, so two configurations cannot be mixed in one file.
    A provider that returns reasoning and response merged (Together) needs stop at the end-of-reasoning token and response_open, the string that opens the response block: each rollout is then two calls. kw goes to complete: max_tokens (covering reasoning plus response), temperature, top_p and top_k (pass them explicitly), timeout, ...
    Which positions to sample and how many is up to the caller: fill({t: count}) tops up the deficit at each position, so a strided grid is one call, and deficit(eps) is the next round of an early-stopping scheme. scores(key, method) is estimate over the rollouts: with reuse (the default) every rollout counts at each position it carries the text through, since a rollout from t whose continuation starts with the text's next k tokens is a sample from t + k as well, and the recursion reads those counts backward from the end so that later rollouts inform earlier positions; either way a stride-1 grid at a small count per position is the grid to use."""

    def __init__(self, tokenizer, prompt: str, text: str, path: str, model: str, provider: str, judge: Callable[[dict], Awaitable[dict]] | None = None, stop: str | None = None, response_open: str = "", **kw):
        self.tok, self.prompt, self.path, self.model, self.provider, self.judge, self.stop, self.response_open, self.kw = tokenizer, prompt, path, model, provider, judge, stop, response_open, kw
        self.judged_path = os.path.splitext(path)[0] + ".judged.jsonl"
        self.prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        self.n_prompt = len(self.prompt_ids)
        full = tokenizer(prompt + text, add_special_tokens=False)["input_ids"]
        if full[:self.n_prompt] != self.prompt_ids:
            raise ValueError(f"text merges into the prompt at the seam: the prompt ends with ids {self.prompt_ids[-2:]} but prompt + text has {full[self.n_prompt - 2:self.n_prompt + 1]} there, so no position cuts cleanly. Render the prompt so its last token cannot merge with the text's first (e.g. a text starting with a newline after a prompt ending in one)")
        self.ids = full[self.n_prompt:]  # the text tokenized after the prompt, as the model produced it
        self.strs = [tokenizer.decode(self.ids[:t]) for t in range(len(self.ids) + 1)]  # the first t tokens of text as a string, for every t
        self.cfg = json.loads(json.dumps({"model": model, "provider": provider, "stop": stop, "response_open": response_open, "kw": kw, "prompt": _sha(prompt), "text": _sha(text)}, default=str))  # stamped on every rollout; round-tripped so it compares equal to a loaded one
        self._prefixes: dict[int, str | None] = {}
        self._reuse: dict[tuple[int, int], int] = {}
        self.rollouts, self.judged = self._load()
        print(f"  {gray}prompt {self.n_prompt} tokens, text {len(self.ids)} tokens, {len(self.rollouts)} rollouts ({_usd(sum(r['cost'] for r in self.rollouts))}, {len(self.judged)} judged) on disk at {path}{endc}")

    def _load(self) -> tuple[list[dict], set[tuple[int, int]]]:
        """The rollouts on disk at path, each checked against this instance: prompt_tokens == n_prompt + t (else a different prompt or text), and its cfg stamp equal to this one where it carries one. A mismatch raises rather than mixing runs; records without a stamp (an older format) are noted. The sidecar's verdicts are merged into the records, and the (t, i) of every judged rollout come back with them."""
        if not os.path.exists(self.path):
            return [], set()
        rollouts = [json.loads(line) for line in open(self.path)]
        unstamped = 0
        for n, r in enumerate(rollouts, 1):
            if r["prompt_tokens"] != self.n_prompt + r["t"]:
                raise ValueError(f"{self.path} line {n}: t={r['t']} with {r['prompt_tokens']} prompt tokens, but this prompt and text give {self.n_prompt + r['t']}: the file holds rollouts of a different prompt or text")
            if "cfg" not in r:
                unstamped += 1
            elif r["cfg"] != self.cfg:
                diff = [k for k in self.cfg if r["cfg"].get(k) != self.cfg[k]]
                raise ValueError(f"{self.path} line {n}: rollout sampled under a different configuration, {' and '.join(diff)} differ: {[r['cfg'].get(k) for k in diff]} on disk vs {[self.cfg[k] for k in diff]} here")
        if unstamped:
            print(f"  {yellow}{unstamped} rollouts in {self.path} carry no configuration stamp (an older format); only their token counts were checked{endc}")
        judged = {(r["t"], r["i"]) for r in rollouts if r.keys() - ROLLOUT_KEYS}
        by_key = {(r["t"], r["i"]): r for r in rollouts}
        for n, line in enumerate(open(self.judged_path) if os.path.exists(self.judged_path) else [], 1):
            v = json.loads(line)
            key = (v.pop("t"), v.pop("i"))
            if key not in by_key:
                raise ValueError(f"{self.judged_path} line {n}: a verdict for rollout {key}, which {self.path} does not hold")
            by_key[key] |= v
            judged.add(key)
        return rollouts, judged

    def prefix(self, t: int) -> str | None:
        """The first t tokens of text as a string, or None when prompt + that string does not retokenize to the prompt's ids followed by those t ids (a cut inside a multi-byte character, or a token that only exists merged with its neighbor). What is sent is prompt + prefix, so that is what is checked. Cached per instance."""
        if t not in self._prefixes:
            self._prefixes[t] = self.strs[t] if self.tok(self.prompt + self.strs[t], add_special_tokens=False)["input_ids"] == self.prompt_ids + self.ids[:t] else None
        return self._prefixes[t]

    def grid(self, stride: int) -> list[int]:
        """Every stride-th position plus 0 and the end."""
        return sorted(set(range(0, len(self.ids) + 1, stride)) | {len(self.ids)})

    async def rollout(self, t: int, i: int, client: httpx.AsyncClient | None = None) -> dict:
        """One continuation from position t, appended to the file and to self.rollouts, not yet judged. With response_open the response is sampled in a second call after the reasoning stops, and stays empty when the reasoning hit max_tokens. Both calls check the provider's prompt_tokens against the local count of the exact string sent."""
        if not 0 <= t <= len(self.ids):
            raise ValueError(f"t={t} is outside the text's {len(self.ids)} tokens")
        if self.prefix(t) is None:
            raise ValueError(f"t={t}: the prefix does not retokenize identically (fill skips such positions)")
        prefix = self.prompt + self.prefix(t)
        body = await complete(prefix, self.model, self.provider, stop=self.stop, client=client, **self.kw)
        r = flat(body)
        if r["prompt_tokens"] != self.n_prompt + t:
            raise RuntimeError(f"t={t}: {r['provider']} counted {r['prompt_tokens']} prompt tokens, expected {self.n_prompt + t}: it wrapped or re-tokenized the prefix")
        text = r["text"] or ""  # a provider sends text: null when nothing was generated, e.g. a trace that used every token
        rec = {"t": t, "i": i, "reasoning": r["reasoning"], "response": text, "finish_reason": r["finish_reason"], "provider": r["provider"], "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"], "cost": r["cost"], "cfg": self.cfg, "raw": [body]}
        if self.response_open:
            rec |= {"reasoning": text, "response": ""}
            if r["finish_reason"] == "stop":
                prefix2 = prefix + text + self.response_open
                body2 = await complete(prefix2, self.model, self.provider, stop=self.stop, client=client, **self.kw)
                r2 = flat(body2)
                n2 = len(self.tok(prefix2, add_special_tokens=False)["input_ids"])
                if r2["prompt_tokens"] != n2:
                    raise RuntimeError(f"t={t}: {r2['provider']} counted {r2['prompt_tokens']} prompt tokens on the response call, expected {n2}: it wrapped or re-tokenized the reasoning")
                rec |= {"response": r2["text"] or "", "finish_reason": r2["finish_reason"], "completion_tokens": rec["completion_tokens"] + r2["completion_tokens"], "cost": rec["cost"] + r2["cost"], "raw": [body, body2]}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.rollouts.append(rec)
        return rec

    async def _judge_one(self, rec: dict) -> dict:
        verdict = await self.judge(rec)
        if clash := verdict.keys() & rec.keys():
            raise ValueError(f"the judge returned {sorted(clash)}, which would overwrite the rollout's own fields; return other names")
        with open(self.judged_path, "a") as f:
            f.write(json.dumps({"t": rec["t"], "i": rec["i"], **verdict}) + "\n")
        rec |= verdict
        self.judged.add((rec["t"], rec["i"]))
        return rec

    async def judge_pending(self, concurrency: int = 32, desc: str = "judge") -> list[dict | None]:
        """Judges every rollout on disk without a verdict, at most concurrency at a time under gather_bar, appending each verdict to the sidecar as it comes. Returns the rollouts judged, None where the judge failed with RequestFailed; the next call tries those again. fill calls this after sampling."""
        if self.judge is None:
            raise ValueError("no judge set")
        todo = [r for r in self.rollouts if (r["t"], r["i"]) not in self.judged]
        if not todo:
            return []
        print(f"  {gray}{len(todo)} rollouts to judge{endc}")
        recs = await gather_bar([self._judge_one(r) for r in todo], concurrency, desc)
        if None in recs:
            print(f"  {yellow}{recs.count(None)} judgments failed (causes in the summary above); another judge_pending or fill call tries them again{endc}")
        return recs

    async def fill(self, want: dict[int, int], concurrency: int = 32, desc: str = "resample") -> list[dict | None]:
        """Samples until each position t in want has want[t] rollouts on disk, at most concurrency at a time under gather_bar, then judges every unjudged rollout on disk (judge_pending) when a judge is set. Positions whose prefix does not retokenize are skipped with a note. Returns the new rollouts, None where sampling failed; calling again samples those and judges whatever the judge failed on. i continues from the largest on disk at t, so (t, i) is unique across fills."""
        skipped = [t for t in want if self.prefix(t) is None]
        if skipped:
            print(f"  {yellow}skipping {len(skipped)} positions whose prefix does not retokenize identically: {skipped[:20]}{endc}")
        done = Counter(r["t"] for r in self.rollouts)
        next_i = {t: max((r["i"] for r in self.rollouts if r["t"] == t), default=-1) + 1 for t in want}
        todo = [(t, next_i[t] + k) for t in want if t not in skipped for k in range(want[t] - done[t])]
        est = f", ~{_usd(len(todo) * sum(r['cost'] for r in self.rollouts) / len(self.rollouts))} at the mean cost so far" if self.rollouts else ""
        print(f"  {gray}{len(todo)} rollouts to sample over {len(want) - len(skipped)} positions, {sum(done[t] for t in want)} already on disk{est}{endc}")
        recs = []
        if todo:
            async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)) as client:
                recs = await gather_bar([self.rollout(t, i, client) for t, i in todo], concurrency, desc)
            if None in recs:
                print(f"  {yellow}{recs.count(None)} rollouts failed (causes in the summary above); another fill call samples them again{endc}")
        if self.judge:
            await self.judge_pending(concurrency)
        return recs

    def reuse(self, r: dict) -> int:
        """How many of the text's tokens after t the rollout r continues with: the k such that prompt + prefix + its continuation (reasoning where the provider split it, else response) tokenizes to the prompt's ids, the prefix's, the text's tokens t+1..t+k, then something else. The rollout is then a sample from every position through t + k, exactly: conditional on producing those tokens, the rest of it is drawn from the model at t + k. Cached per rollout; scores tokenizes the uncached ones in one batch."""
        if (r["t"], r["i"]) not in self._reuse:
            self._reuse_fill([r])
        return self._reuse[r["t"], r["i"]]

    def _reuse_fill(self, rollouts: list[dict]):
        trace = {"prompt": self.prompt, "prompt_ids": self.prompt_ids, "ids": self.ids}
        for r, ids in zip(rollouts, joint_ids(self.tok, trace, [(r["t"], r["reasoning"] if r["reasoning"] is not None else r["response"]) for r in rollouts])):
            self._reuse[r["t"], r["i"]] = 0 if ids is None else k_ids(ids, self.ids, r["t"])

    def scores(self, key: str = "match", method: str = "reuse") -> list[dict]:
        """estimate over the rollouts on disk, with the outcome read from key (a verdict field; None or missing, so unjudged rollouts too, counts as other), plus each position's token (the one ending the prefix). With reuse or the recursion, a rollout counts at every position through t + self.reuse(rollout), so n is the effective count and positions never sampled directly appear, including those fill skips."""
        if method != "naive":
            self._reuse_fill([r for r in self.rollouts if (r["t"], r["i"]) not in self._reuse])
        if pending := sum((r["t"], r["i"]) not in self.judged for r in self.rollouts):
            print(f"  {yellow}{pending} rollouts are not judged yet and count as other{endc}")
        out = estimate(len(self.ids), [(r["t"], self.reuse(r) if method != "naive" else 0, r.get(key)) for r in self.rollouts], method)
        for s in out:
            s["token"] = self.tok.decode(self.ids[s["t"] - 1:s["t"]]) if s["t"] else ""
        return out

    def deficit(self, eps: float, key: str = "match", positions: list[int] | None = None, batch: int = 5, n_min: int = 0, n_max: int | None = None) -> dict[int, int]:
        """What to fill next so that every position's estimate is tight enough: {t: rollouts wanted at t} over positions (default every one that can be sampled), asking batch more where the Wilson half-width of p is above eps, or min(batch, what the floor lacks) while fewer than n_min samples are judged there, and nothing once n samples reach n_max (all counted with reuse). Empty when every position is settled, so `while d := rs.deficit(0.1, n_min=20): await rs.fill(d)` is the early-stopping loop, a round at a time so that rollouts drawn early in the text cover later positions before those are asked for. Raises while rollouts are unjudged (judge_pending first), since it would ask for replacements for them. The interval alone reads rare outcomes as dead (at p = 0.05, a batch of 5 without a hit settles the position at about 16 samples), so pair eps with a floor of about 3 / (the smallest p to resolve)."""
        if pending := sum((r["t"], r["i"]) not in self.judged for r in self.rollouts):
            raise RuntimeError(f"{pending} rollouts are not judged yet: judge_pending() first, or deficit would ask for replacements for them")
        sc = {s["t"]: s for s in self.scores(key)}
        out = {}
        for t in positions if positions is not None else range(len(self.ids) + 1):
            s = sc.get(t, {"n": 0, "direct": 0, "judged": 0, "ci": (0.0, 1.0)})
            extra = min(batch, n_min - s["judged"]) if s["judged"] < n_min else batch if s["ci"][1] - s["ci"][0] > 2 * eps else 0
            if extra and (n_max is None or s["n"] < n_max) and self.prefix(t) is not None:
                out[t] = s["direct"] + extra
        return out

def load_rollouts(trace: dict, path: str, tokenizer) -> list[dict]:
    """The rollouts of one source of a trace, sampled locally or through a provider, in one form, so that analysis is written once. trace holds prompt (the rendered string), prompt_ids, ids (the trace's tokens after the prompt) and cap (the most tokens a response may hold, prefix plus continuation, or None). path is a jsonl with one rollout per line, t (trace tokens kept) and i unique in the file, in either shape: local, with ids (the continuation's sampled tokens) and ended (an eos came before the cap), or a Resampler record. Any other field is a verdict, and the sidecar next to the file (.judged.jsonl in place of .jsonl) adds verdicts keyed by (t, i).
    Per rollout: t, i, ids (as sampled; for a Resampler record joint_ids of the text, or the text tokenized alone where it merged into the prefix), text (the continuation: the decoded ids, or reasoning where the record holds one, else response), stopped (ended, or finish_reason "stop"), capped (not stopped, or t + len(ids) >= cap: one rule for both sources, since Resampler counts max_tokens from the cut and the local samplers cap the total length, and reuse needs a rollout's distribution not to depend on its cut), k (the reuse count: k_ids of sampled ids, else of the joint ids, 0 where they merged), tokens (len(ids) plus one for a stop, or the provider's completion_tokens), the verdict fields, and raw (the stored record)."""
    recs = [json.loads(line) for line in open(path)]
    verdicts = {(r["t"], r["i"]): {f: v for f, v in r.items() if f not in ROLLOUT_KEYS | {"ids", "ended"}} for r in recs}
    if len(verdicts) != len(recs):
        raise ValueError(f"{path}: {len(recs) - len(verdicts)} rollouts repeat the (t, i) of another")
    side = os.path.splitext(path)[0] + ".judged.jsonl"
    for n, line in enumerate(open(side) if os.path.exists(side) else [], 1):
        v = json.loads(line)
        key = (v.pop("t"), v.pop("i"))
        if key not in verdicts:
            raise ValueError(f"{side} line {n}: a verdict for rollout {key}, which {path} does not hold")
        verdicts[key] |= v
    if bad := [(r["t"], r["i"]) for r in recs if "ids" not in r and "response" not in r]:
        raise ValueError(f"{path}: rollouts {bad[:3]} hold neither ids nor response")
    api = [j for j, r in enumerate(recs) if "ids" not in r]
    text = {j: recs[j]["reasoning"] if recs[j].get("reasoning") is not None else recs[j]["response"] for j in api}
    joint = dict(zip(api, joint_ids(tokenizer, trace, [(recs[j]["t"], text[j]) for j in api])))
    out = []
    for j, r in enumerate(recs):
        if "ids" in r:
            row = {"ids": r["ids"], "text": tokenizer.decode(r["ids"]), "stopped": r["ended"], "k": k_ids(r["ids"], trace["ids"], r["t"]), "tokens": len(r["ids"]) + r["ended"]}
        else:
            row = {"ids": joint[j] if joint[j] is not None else tokenizer(text[j], add_special_tokens=False)["input_ids"], "text": text[j], "stopped": r["finish_reason"] == "stop", "k": 0 if joint[j] is None else k_ids(joint[j], trace["ids"], r["t"]), "tokens": r["completion_tokens"]}
        row = {"t": r["t"], "i": r["i"], **row, "capped": not row["stopped"] or (trace["cap"] is not None and r["t"] + len(row["ids"]) >= trace["cap"]), "raw": r}
        if clash := row.keys() & verdicts[r["t"], r["i"]].keys():
            raise ValueError(f"{path}: rollout {(r['t'], r['i'])} has verdict fields named like the reader's own: {sorted(clash)}")
        out.append(row | verdicts[r["t"], r["i"]])
    return out

def rollout_scores(trace: dict, rollouts: list[dict], tokenizer, key="match", method: str = "reuse") -> list[dict]:
    """estimate over load_rollouts' output, as Resampler.scores gives for its own file: the outcome is the verdict field key, or key(rollout) where key is a function (an outcome computed from the text, which then applies the cap rule itself: lambda r: not r["capped"] and ...), and each position carries its token."""
    out = estimate(len(trace["ids"]), [(r["t"], r["k"], key(r) if callable(key) else r.get(key)) for r in rollouts], method)
    for s in out:
        s["token"] = tokenizer.decode(trace["ids"][s["t"] - 1:s["t"]]) if s["t"] else ""
    return out

PROBE_MSGS = [{"role": "user", "content": "What is 17 * 23?"}]
PROBE_COT = "Okay, 17 * 23. 17 * 20 = 340, 17 * 3 = 51, so 340 + 51 ="  # ends on a token boundary; a model that sees the open think block continues with " 391"
PROBE_WS = "Okay.\t\tLet me see...\n\n\n\n17 * 23 = "  # tabs and blank lines: a provider that normalizes whitespace changes this count

async def _probe_one(tok, model: str, ep: dict, strings: list[str], samples: int, long: int, client: httpx.AsyncClient) -> dict:
    n = lambda s: len(tok(s or "", add_special_tokens=False)["input_ids"])
    pricing = ep.get("pricing") or {}
    row = {"provider": ep["provider"], "quant": ep.get("quantization"), "$/M in,out": ", ".join(f"{float(pricing[k]) * 1e6:.2f}" if k in pricing else "?" for k in ("prompt", "completion"))}
    try:
        rs = [flat(await complete(s, model, ep["provider"], max_tokens=1, attempts=4, client=client)) for s in strings]
        row["offsets"] = [r["prompt_tokens"] - n(s) for r, s in zip(rs, strings)]
        if any(row["offsets"]):
            row["verdict"] = ("wrapped" + (" + system prompt" if min(row["offsets"]) > 40 else "")) if max(row["offsets"]) - min(row["offsets"]) <= 2 else "re-tokenized"
            return row
        rs = [flat(b) for b in await asyncio.gather(*[complete(strings[1], model, ep["provider"], max_tokens=12, attempts=4, client=client) for _ in range(samples)])]
        conts = [(r["reasoning"] or "") + (r["text"] or "") for r in rs]
        row["continues"] = f"{sum(c.lstrip().startswith('391') for c in conts)}/{samples}"
        row["field"] = "+".join(k for k in ("reasoning", "text") if any(r[k] for r in rs))
        row["sample"] = next((c for c in conts if not c.lstrip().startswith("391")), conts[0])[:40]
        r = flat(await complete(strings[0], model, ep["provider"], max_tokens=long, attempts=4, client=client))
        row |= {"finish": r["finish_reason"], "special": [s for s in tok.all_special_tokens if s in (r["reasoning"] or "") + (r["text"] or "")], "extra_toks": r["completion_tokens"] - n(r["reasoning"]) - n(r["text"]), "reasoning_toks": f"{r['reasoning_tokens']} vs {n(r['reasoning'])} local"}
        row["verdict"] = (("pass" if row["continues"] == f"{samples}/{samples}" else "restarts") + ("" if "reasoning" in row["field"] else ", merged: needs stop + response_open")
                          + (", leaks special tokens" if row["special"] else "") + (", extra_toks off" if row["finish"] == "stop" and not 0 <= row["extra_toks"] <= 8 else ""))
    except RequestFailed as e:
        row["verdict"] = f"{'rejected' if e.why[0] == '4' and e.why not in ('408', '429') else 'unreachable'} {e.why}"
    return row

async def probe(tokenizer, model: str, render: Callable[[list[dict]], str], providers: list[str] | None = None, samples: int = 8, long: int = 400, extra: tuple[str, ...] = (), concurrency: int = 16) -> list[dict]:
    """Checks every endpoint serving model (or just providers) for raw-prompt passthrough, a few cents in total. render(messages) is the project's prompt renderer and must end inside the open think block; the probe renders one arithmetic question and sends, per endpoint:
    1. max_tokens=1 requests for the bare prompt, the prompt plus a partial CoT ending "340 + 51 =", the prompt plus whitespace-heavy text, and each full prompt string in extra (a real prompt plus its full-length trace, or a two-stage provider's prefix + reasoning + response_open). offsets = provider prompt_tokens minus local count per string; any nonzero offset fails (nearly constant: chat wrapper, above 40: with an injected system prompt, varying widely: a different tokenizer).
    2. samples continuations of the partial CoT at max_tokens=12. continues counts those starting with 391; the rest restarted (the rendering is wrong, or the block was closed), and sample shows one. field is where they arrived: reasoning (the provider splits at the think close) or text (merged: use stop and response_open).
    3. one generation of up to long tokens from the bare prompt: finish, special tokens leaked into the output, extra_toks = completion_tokens minus the local tokens of reasoning + text (the provider's constant for the record check, meaningful when finish is stop: raise long if it is length), and reported vs local reasoning tokens. The verdict flags leaked special tokens, and an extra_toks outside 0..8 when finish is stop (below 0: the provider counts fewer tokens than the text holds, so a different tokenizer; above 8: tokens hidden from the returned text).
    A 4xx (rejected: not served raw) or exhausted retries (unreachable: throttled or down, try again later) is the verdict. Prints the table and returns one row per endpoint."""
    p0 = render(PROBE_MSGS)
    strings = [p0, p0 + PROBE_COT, p0 + PROBE_WS, *extra]
    eps = [e for e in endpoints(model) if providers is None or e["provider"] in providers]
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency * samples, max_keepalive_connections=concurrency * samples)) as client:
        rows = await gather_bar([_probe_one(tokenizer, model, e, strings, samples, long, client) for e in eps], concurrency, f"probe {model}")
    headers = ["provider", "quant", "$/M in,out", "offsets", "continues", "field", "sample", "finish", "special", "extra_toks", "reasoning_toks", "verdict"]
    show_table(headers, [tuple(str(r.get(h, "")) for h in headers) for r in rows], title=f"{model} raw /completions probe on {len(rows)} endpoints")
    return rows

async def sampling_defaults(model: str, provider: str, prompt: str, n: int = 500, concurrency: int = 32) -> dict[str, Counter]:
    """n single-token draws at temperature 1 from prompt, which should sit at a high-entropy position with no think block open (a bare "Once upon a time, in a land far away, there lived a" on a passing provider): first with top_p and top_k omitted, then with top_p=1.0, top_k=0 explicit. A provider that applies the checkpoint's generation_config when they are omitted shows far fewer distinct tokens in the first run (Qwen3.6-27B on CoreWeave: 4 vs 9 to 13 in 500 draws; Chutes and Phala: no difference); one that ignores the explicit values shows the same clipped tail both ways. Two runs of 500 differ by about 0.1 in total variation from noise alone. Returns both Counters."""
    counts = {}
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)) as client:
        for name, kw in {"omitted": {}, "explicit": {"top_p": 1.0, "top_k": 0}}.items():
            rs = [flat(b) for b in await gather_bar([complete(prompt, model, provider, max_tokens=1, client=client, **kw) for _ in range(n)], concurrency, name) if b]
            counts[name] = Counter((r["reasoning"] or r["text"] or "") for r in rs)
            print(f"  {name}: {len(counts[name])} distinct tokens in {sum(counts[name].values())} draws; top {counts[name].most_common(5)}, tail {counts[name].most_common()[-5:]}")
    return counts

def calib_rollouts(model, trace: dict, rollouts: list[dict], batch_size: int = 8, quiet: bool = False) -> list[dict]:
    """Every sampled token of every rollout (load_rollouts' output) against the local model, for calib_check: do a provider's rollouts look like samples from this model? Each rollout runs through one uncached forward as prompt ids + the trace's first t ids + its ids, in right-padded batches sorted by length; the forward returns logits for batch x length x vocabulary, which sets batch_size (14 GiB at 24 rows of 1,200 tokens on a 248k vocabulary). The model's eos ids are merged into one stop token at every step, since a rollout does not record which of them ended it, and the stop that ended a rollout is scored as its last token.
    Per rollout: t, i, n (tokens scored), and sums over its steps, with p the model's distribution at a step and x the token sampled there: s of log p(x) + H(p), which has mean 0 for x drawn from p, and v of its variance Var_p[log p]; rare as [seen, expected, variance] of tokens with p(x) < 0.01; stop as [1 if the rollout stopped, the summed stop probability, its variance]. A record with reasoning split off from the response raises: its sampled sequence is not prompt + prefix + one text. quiet hides the progress bar."""
    if bad := [(r["t"], r["i"]) for r in rollouts if r["raw"].get("reasoning")]:
        raise ValueError(f"rollouts {bad[:3]} hold reasoning apart from the response, which calib_rollouts cannot score")
    eos, dev, n_prompt, out = sorted(eos_ids(model)), next(model.parameters()).device, len(trace["prompt_ids"]), []
    rollouts = sorted(rollouts, key=lambda r: r["t"] + len(r["ids"]))
    for b in pbar(range(0, len(rollouts), batch_size), desc="calib", disable=quiet):
        batch = rollouts[b:b + batch_size]
        seqs = [trace["prompt_ids"] + trace["ids"][:r["t"]] + r["ids"] for r in batch]
        logits = model(t.tensor([q + [0] * (len(seqs[-1]) - len(q)) for q in seqs], device=dev), return_type="logits")  # right padding: nothing a row is scored on can see it
        for row, r in zip(logits, batch):
            x, first, acc = t.tensor(r["ids"] + eos[-1:] * r["stopped"], device=dev), n_prompt + r["t"] - 1, t.zeros(7, dtype=t.float64)
            for c in range(0, len(x), 100):  # 100 steps at a time, to bound the [steps, vocabulary] temporaries
                xc = x[c:c + 100]
                lp = row[first + c:first + c + len(xc)].float().log_softmax(-1)
                lp[:, eos[-1]] = lp[:, eos].logsumexp(-1)  # the stop token: all eos ids as one
                lp[:, eos[:-1]] = -1e9
                p, lx = lp.exp(), lp[t.arange(len(xc)), xc]
                H, a, ps = -(p * lp).sum(-1), (p * (p < 0.01)).sum(-1), p[:, eos[-1]]
                acc += t.stack([(lx + H).sum(), ((p * lp * lp).sum(-1) - H * H).sum(), (lx < math.log(0.01)).sum(), a.sum(), (a * (1 - a)).sum(), ps.sum(), (ps * (1 - ps)).sum()]).double().cpu()
            s, v, *rest = acc.tolist()
            out.append({"t": r["t"], "i": r["i"], "n": len(x), "s": s, "v": v, "rare": rest[:3], "stop": [float(r["stopped"]), *rest[3:]]})
    return out

def calib_check(calib: list[dict], z_max: float = 3.0) -> dict:
    """The fungibility check over calib_rollouts' output for one source: can its rollouts stand in for the local model's? Three z-scores, each standard normal for samples from the local model: tokens (sum s over the root of sum v; negative when the tokens are less probable than the model's own samples would be: a higher temperature, another model), rare (tokens under 1%, seen minus expected; negative under top-p or top-k truncation) and stops (seen minus expected; positive when the source ends responses where the model would not). flagged when any |z| exceeds z_max, which samples from the model meet about 0.8% of the time at 3. Also rollouts, tokens, mean_s (nats per token) and temperature, the temperature at which the model fits the tokens best, by one Newton step from 1.
    Use: probe first, then about 150 rollouts through the provider, load_rollouts, calib_rollouts, calib_check. Measured on Llama-3.1-8B, Llama-3.2-3B, Qwen3.5-9B and Qwen3.8-27B against 11 provider deployments and 3 altered-sampling controls, at 150 rollouts: local rollouts are flagged 0.2 to 0.8% of the time, the 8 deployments whose P(correct) curve matches local at most 2%, every source 0.048 or more off on the curve at least 99.6%, and top-k and top-p truncation always.
    Limits. A flag says the source does not sample as the local model does, not that an outcome differs: truncation is flagged and leaves P(correct) where it was, and the size of mean_s does not predict the size of an outcome gap. A pass at 150 rollouts does not rule out a small difference (a deployment 0.034 off on the curve passed at 150 and showed at 3,100), nor a shift in next-token probability at a single position (two deployments were 0.05 off at one close call with matching curves and scores: a branch an analysis depends on is tested with one-token samples there). Repeated or cached samples look like fresh ones. The evidence is one trace per model, with thinking off. ids recovered from a provider's text are not always the sampled ones (joint_ids), which did not move the scores of matching deployments."""
    tot = lambda f: sum(f(c) for c in calib)
    z = lambda d, var: d / math.sqrt(var) if var > 0 else math.copysign(math.inf, d) if d else 0.0
    s, v, n = tot(lambda c: c["s"]), tot(lambda c: c["v"]), tot(lambda c: c["n"])
    rare, stop = [tot(lambda c, j=j: c["rare"][j]) for j in range(3)], [tot(lambda c, j=j: c["stop"][j]) for j in range(3)]
    zs = {"tokens": z(s, v), "rare": z(rare[0] - rare[1], rare[2]), "stops": z(stop[0] - stop[1], stop[2])}
    return {"rollouts": len(calib), "tokens": n, "mean_s": s / n, "temperature": 1 + s / (2 * s - v), "z": zs, "flagged": any(abs(x) > z_max for x in zs.values())}

def resample_curve(scores: list[dict] | dict[str, list[dict]], title: str = "", renderer=None, return_fig: bool = False):
    """p over t with its interval as a band, from Resampler.scores, rollout_scores or estimate. A dict of named score lists draws one curve each (the methods, or the sources of one trace): four hues, then the same four dashed, and more than eight raises. The band is clipped to [0, 1], which the recursion's interval can leave. Hovering a point shows its counts and interval, and its token where the scores carry one."""
    named = scores if isinstance(scores, dict) else {"": scores}
    if len(named) > 2 * len(SERIES):
        raise ValueError(f"{len(named)} curves, but there are {2 * len(SERIES)} styles: draw fewer per figure")
    fig = go.Figure()
    for j, (name, sc) in enumerate(named.items()):
        color, ts = SERIES[j % len(SERIES)], [s["t"] for s in sc]
        lo, hi = [max(s["ci"][0], 0.0) for s in sc], [min(s["ci"][1], 1.0) for s in sc]
        hover = [(f"{name}: " if name else "") + f"t={s['t']}" + (f" {s['token']!r}" if "token" in s else "") + f"<br>p={s['p']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]<br>{s['k']}/{s['judged']} judged, {s['other']} other, {s['direct']} of {s['n']} sampled here" for s in sc]
        fig.add_scatter(x=ts + ts[::-1], y=hi + lo[::-1], fill="toself", fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.2)", line={"width": 0}, hoverinfo="skip", showlegend=False, legendgroup=name)
        fig.add_scatter(x=ts, y=[s["p"] for s in sc], mode="lines+markers", line={"color": color, "dash": "solid" if j < len(SERIES) else "dash"}, text=hover, hoverinfo="text", name=name, legendgroup=name, showlegend=bool(name))
    fig.update_layout(title=title, xaxis_title="prefix tokens t", yaxis_title="p", yaxis_range=[0, 1], **DARK)
    return fig if return_fig else fig.show(renderer=renderer)
