"""Token-level resampling through OpenRouter's raw /completions: cut a CoT or a response at token positions, sample continuations per prefix into a rollouts jsonl that reruns top up per position, judge them, and score P(outcome | prefix_t) with Wilson intervals.

The model continues the CoT only if the provider feeds it exactly the rendered prompt string, and nothing in a response says whether it did. Before a paid run, in this order:

1. Render the prompt so it ends inside the open think block (or the open text block, to cut a reasoning-off response). Qwen: apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True) ends "<|im_start|>assistant\\n<think>\\n". DeepSeek V4 ships encoding/encoding_dsv4.py instead of a template (hf_hub_download it, sys.path.insert its directory); encode_messages(msgs, "thinking") ends "<｜Assistant｜><think>". Inkling's template ends at <|message_model|>, so append <|content_thinking|> (or <|content_text|> for a response cut).
   Chat-message prefill, a trailing assistant message holding the partial CoT, never continues it: every template closes the block, the model starts over or answers, and some providers render EOS so the model emits garbage. Count tokens with add_special_tokens=False and know whether the tokenizer adds BOS on its own, since the provider's count is compared with the local one.
2. probe(tokenizer, model, render) on every endpoint, a few cents. Pass = prompt_tokens exactly equal to the local count on every string, and every continuation of "... 340 + 51 =" starting with 391. Failures: an offset of a few tokens, constant give or take one from tokens merging at the wrapper's boundaries, is a chat wrapper (the model answers from scratch: "The product of 17 and 23 is 391."); about +80 is an injected system prompt on top; offsets that vary widely mean a different tokenizer; a count that varies between identical requests is a heterogeneous backend; a 4xx means the endpoint is not served raw, and exhausted retries mean it was throttled or down at probe time. Exact counts with restarts ("Here's a thinking process:", "The user wants...") mean the model did not see an open think block: the local rendering is wrong, and nothing at run time catches that.
   Continuations arriving in text rather than reasoning mean the provider returns reasoning and response merged with the special tokens stripped (Together): set stop to the end-of-reasoning token and response_open to the string that opens the response block, and every rollout is two calls. Special tokens in returned text mean this provider does not strip them. Rosters change: on 2026-09-19 Qwen3.6-27B passed on Chutes and Phala and DeepSeek V4 Flash on Parasail and Mancer, while CoreWeave, which passed for both on 2026-09-02, no longer served either.
3. Resample on the endpoint that produced the base rollouts, or accept that the curve measures the resampling endpoint's deployment: quantization (fp8, fp4, unlisted) and sampling defaults differ. Pass top_p=1.0 and top_k=0 explicitly: CoreWeave applied Qwen's generation_config (top_k 20, top_p 0.95) when they were omitted, Chutes and Phala did not; sampling_defaults measures this. seed is honored by some providers and ignored by others. Cross-provider curves agreed within Wilson intervals at S=50; base-rate differences below that resolution went undetected.
4. max_tokens must cover reasoning plus response: Inkling traces exhausted 8192 and produced empty responses that looked like throttling. Such rollouts finish with "length", count as other, and are not retried. A few-hundred-token trace at S=50 stride 1 costs tens of dollars on a $2/M model, and a judge costs about as much as the subject on cheap ones.
5. At run time every rollout checks prompt_tokens == n_prompt + t (and the response call's count on a two-stage provider) and raises otherwise, catching wrapping, re-tokenization, or a provider change mid-run; finish_reason "error" (a mid-stream abort whose usage is wrong too) is retried inside complete. In the project, check the base record against the same tokenizer: len(tokenizer(prompt)) == its prompt_tokens (plus a known constant, e.g. Inkling's appended block), and its completion_tokens minus the local tokens of reasoning + response equal to the provider's constant (1 or 2 for DeepSeek and Qwen: closing tag and EOS; 5 to 7 for Inkling), a larger gap being a truncated trace. usage reasoning_tokens is 0 or wrong on many providers: count from text.
   Cuts inside a multi-byte character (byte-level BPE) do not retokenize and are skipped. Providers throttle in bursts (DeepInfra 429s independent of request rate, Together 503s above ~12 concurrent two-stage rollouts): keep concurrency at 12 to 32 and call fill again. Prompt logprobs and echo are unavailable on the raw endpoint, so prompt identity rests on counts plus the continuation check. Closed-lab models return summarized or encrypted reasoning and cannot be resampled."""

import asyncio
import functools
import hashlib
import json
import os
from collections import Counter
from collections.abc import Awaitable, Callable

import httpx
import plotly.graph_objects as go

from mechtools.colors import *
from mechtools.openrouter import RequestFailed, _usd, complete, endpoints, flat, gather_bar
from mechtools.stats import wilson
from mechtools.tables import show_table

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]

class Resampler:
    """Resamples text (a CoT, or a reasoning-off response) from token position t: the provider continues prompt + the first t tokens of text through raw /completions, so prompt must be the rendered chat template ending inside the open think or text block, and provider must pass raw prompts through verbatim: run probe first, and read the module docstring for the rest of the checklist. text is tokenized after the prompt, as the model produced it, and must not merge into the prompt's last token. Each rollout checks the provider's prompt_tokens against the local count of the exact string sent (on the response call too) and raises otherwise.
    Rollouts append to the jsonl at path, one per line: t, i, reasoning, response, finish_reason, provider, prompt_tokens, completion_tokens, cost, cfg (the configuration stamp: model, provider, stop, response_open, kw, hashes of prompt and text), raw (the response bodies, one per call), plus whatever judge(rollout) returns, which may not reuse those names. response is the continuation only; when text is a response, the judge sees the whole thing as prefix(t) + rollout["response"]. The judge is called on every rollout, including empty and truncated ones, and decides what to return for them.
    Loading a file checks every record's token counts and stamp against the instance and raises on a mismatch, so two configurations cannot be mixed in one file.
    A provider that returns reasoning and response merged (Together) needs stop at the end-of-reasoning token and response_open, the string that opens the response block: each rollout is then two calls. kw goes to complete: max_tokens (covering reasoning plus response), temperature, top_p and top_k (pass them explicitly), timeout, ...
    Which positions to sample and how many is up to the caller: fill({t: count}) tops up the deficit at each position, so a strided grid is one call and an adaptive scheme is a loop over fill and scores."""

    def __init__(self, tokenizer, prompt: str, text: str, path: str, model: str, provider: str, judge: Callable[[dict], Awaitable[dict]] | None = None, stop: str | None = None, response_open: str = "", **kw):
        self.tok, self.prompt, self.path, self.model, self.provider, self.judge, self.stop, self.response_open, self.kw = tokenizer, prompt, path, model, provider, judge, stop, response_open, kw
        self.prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        self.n_prompt = len(self.prompt_ids)
        full = tokenizer(prompt + text, add_special_tokens=False)["input_ids"]
        if full[:self.n_prompt] != self.prompt_ids:
            raise ValueError(f"text merges into the prompt at the seam: the prompt ends with ids {self.prompt_ids[-2:]} but prompt + text has {full[self.n_prompt - 2:self.n_prompt + 1]} there, so no position cuts cleanly. Render the prompt so its last token cannot merge with the text's first (e.g. a text starting with a newline after a prompt ending in one)")
        self.ids = full[self.n_prompt:]  # the text tokenized after the prompt, as the model produced it
        self.cfg = json.loads(json.dumps({"model": model, "provider": provider, "stop": stop, "response_open": response_open, "kw": kw, "prompt": _sha(prompt), "text": _sha(text)}, default=str))  # stamped on every rollout; round-tripped so it compares equal to a loaded one
        self.rollouts = self._load()
        print(f"  {gray}prompt {self.n_prompt} tokens, text {len(self.ids)} tokens, {len(self.rollouts)} rollouts ({_usd(sum(r['cost'] for r in self.rollouts))}) on disk at {path}{endc}")

    def _load(self) -> list[dict]:
        """The rollouts on disk at path, each checked against this instance: prompt_tokens == n_prompt + t (else a different prompt or text), and its cfg stamp equal to this one where it carries one. A mismatch raises rather than mixing runs; records without a stamp (an older format) are noted."""
        if not os.path.exists(self.path):
            return []
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
        return rollouts

    @functools.cache
    def prefix(self, t: int) -> str | None:
        """The first t tokens of text as a string, or None when prompt + that string does not retokenize to the prompt's ids followed by those t ids (a cut inside a multi-byte character, or a token that only exists merged with its neighbor). What is sent is prompt + prefix, so that is what is checked."""
        s = self.tok.decode(self.ids[:t])
        return s if self.tok(self.prompt + s, add_special_tokens=False)["input_ids"] == self.prompt_ids + self.ids[:t] else None

    def grid(self, stride: int) -> list[int]:
        """Every stride-th position plus 0 and the end."""
        return sorted(set(range(0, len(self.ids) + 1, stride)) | {len(self.ids)})

    async def rollout(self, t: int, i: int, client: httpx.AsyncClient | None = None) -> dict:
        """One continuation from position t, judged, appended to the file and to self.rollouts. With response_open the response is sampled in a second call after the reasoning stops, and stays empty when the reasoning hit max_tokens. Both calls check the provider's prompt_tokens against the local count of the exact string sent."""
        if not 0 <= t <= len(self.ids):
            raise ValueError(f"t={t} is outside the text's {len(self.ids)} tokens")
        if self.prefix(t) is None:
            raise ValueError(f"t={t}: the prefix does not retokenize identically (fill skips such positions)")
        prefix = self.prompt + self.prefix(t)
        body = await complete(prefix, self.model, self.provider, stop=self.stop, client=client, **self.kw)
        r = flat(body)
        if r["prompt_tokens"] != self.n_prompt + t:
            raise RuntimeError(f"t={t}: {r['provider']} counted {r['prompt_tokens']} prompt tokens, expected {self.n_prompt + t}: it wrapped or re-tokenized the prefix")
        rec = {"t": t, "i": i, "reasoning": r["reasoning"], "response": r["text"], "finish_reason": r["finish_reason"], "provider": r["provider"], "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"], "cost": r["cost"], "cfg": self.cfg, "raw": [body]}
        if self.response_open:
            rec |= {"reasoning": r["text"], "response": ""}
            if r["finish_reason"] == "stop":
                prefix2 = prefix + r["text"] + self.response_open
                body2 = await complete(prefix2, self.model, self.provider, stop=self.stop, client=client, **self.kw)
                r2 = flat(body2)
                n2 = len(self.tok(prefix2, add_special_tokens=False)["input_ids"])
                if r2["prompt_tokens"] != n2:
                    raise RuntimeError(f"t={t}: {r2['provider']} counted {r2['prompt_tokens']} prompt tokens on the response call, expected {n2}: it wrapped or re-tokenized the reasoning")
                rec |= {"response": r2["text"], "finish_reason": r2["finish_reason"], "completion_tokens": rec["completion_tokens"] + r2["completion_tokens"], "cost": rec["cost"] + r2["cost"], "raw": [body, body2]}
        if self.judge:
            verdict = await self.judge(rec)
            if clash := verdict.keys() & rec.keys():
                raise ValueError(f"the judge returned {sorted(clash)}, which would overwrite the rollout's own fields; return other names")
            rec |= verdict
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.rollouts.append(rec)
        return rec

    async def fill(self, want: dict[int, int], concurrency: int = 32, desc: str = "resample") -> list[dict | None]:
        """Samples until each position t in want has want[t] rollouts on disk, at most concurrency at a time under gather_bar. Positions whose prefix does not retokenize are skipped with a note. Returns the new rollouts, None where one failed; calling again samples those. i continues from the largest on disk at t, so (t, i) is unique across fills."""
        skipped = [t for t in want if self.prefix(t) is None]
        if skipped:
            print(f"  {yellow}skipping {len(skipped)} positions whose prefix does not retokenize identically: {skipped[:20]}{endc}")
        done = Counter(r["t"] for r in self.rollouts)
        next_i = {t: max((r["i"] for r in self.rollouts if r["t"] == t), default=-1) + 1 for t in want}
        todo = [(t, next_i[t] + k) for t in want if t not in skipped for k in range(want[t] - done[t])]
        est = f", ~{_usd(len(todo) * sum(r['cost'] for r in self.rollouts) / len(self.rollouts))} at the mean cost so far" if self.rollouts else ""
        print(f"  {gray}{len(todo)} rollouts to sample over {len(want) - len(skipped)} positions, {sum(done[t] for t in want)} already on disk{est}{endc}")
        if not todo:
            return []
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)) as client:
            recs = await gather_bar([self.rollout(t, i, client) for t, i in todo], concurrency, desc)
        if None in recs:
            print(f"  {yellow}{recs.count(None)} rollouts failed (causes in the summary above); another fill call samples them again{endc}")
        return recs

    def scores(self, key: str = "match") -> list[dict]:
        """Per position with rollouts, sorted by t: n rollouts, k with key True, judged (True or False), other (None or missing), p = k / judged, ci (Wilson), token (the one ending the prefix)."""
        out = []
        for t in sorted({r["t"] for r in self.rollouts}):
            vals = [r.get(key) for r in self.rollouts if r["t"] == t]
            k, judged = vals.count(True), vals.count(True) + vals.count(False)
            out.append({"t": t, "token": self.tok.decode(self.ids[t - 1:t]) if t else "", "n": len(vals), "k": k, "judged": judged, "other": len(vals) - judged, "p": k / judged if judged else float("nan"), "ci": wilson(k, judged)})
        return out

PROBE_MSGS = [{"role": "user", "content": "What is 17 * 23?"}]
PROBE_COT = "Okay, 17 * 23. 17 * 20 = 340, 17 * 3 = 51, so 340 + 51 ="  # ends on a token boundary; a model that sees the open think block continues with " 391"
PROBE_WS = "Okay.\t\tLet me see...\n\n\n\n17 * 23 = "  # tabs and blank lines: a provider that normalizes whitespace changes this count

async def _probe_one(tok, model: str, ep: dict, strings: list[str], samples: int, long: int, client: httpx.AsyncClient) -> dict:
    n = lambda s: len(tok(s or "", add_special_tokens=False)["input_ids"])
    row = {"provider": ep["provider"], "quant": ep.get("quantization"), "$/M in,out": f"{float(ep['pricing']['prompt']) * 1e6:.2f}, {float(ep['pricing']['completion']) * 1e6:.2f}"}
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

def resample_curve(scores: list[dict], title: str = "", renderer=None, return_fig: bool = False):
    """p over t from Resampler.scores with the Wilson band; hovering a point shows its token, counts and interval."""
    ts, p, lo, hi = [s["t"] for s in scores], [s["p"] for s in scores], [s["ci"][0] for s in scores], [s["ci"][1] for s in scores]
    hover = [f"t={s['t']} {s['token']!r}<br>p={s['p']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]<br>{s['k']}/{s['judged']} judged, {s['other']} other" for s in scores]
    fig = go.Figure([go.Scatter(x=ts + ts[::-1], y=hi + lo[::-1], fill="toself", fillcolor="rgba(31,119,180,0.2)", line={"width": 0}, hoverinfo="skip", showlegend=False),
                     go.Scatter(x=ts, y=p, mode="lines+markers", line={"color": "rgb(31,119,180)"}, text=hover, hoverinfo="text", showlegend=False)])
    fig.update_layout(title=title, xaxis_title="prefix tokens t", yaxis_title="p", yaxis_range=[0, 1], template="plotly_white")
    return fig if return_fig else fig.show(renderer=renderer)
