"""OpenRouter sampling: async chat and raw-completions requests with retries, a bounded gather under an informative progress bar, and the endpoints serving a model. Needs OPENROUTER_API_KEY, which importing mechtools loads from the first .env found walking up from the cwd."""

import asyncio
import contextlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field

import httpx
from dotenv import find_dotenv
from tqdm import tqdm

from mechtools.colors import *

OPENROUTER_URL = "https://openrouter.ai/api/v1"


class RequestFailed(Exception):
    """A request that used up its attempts, or got a 4xx that retrying cannot fix. why is the short cause the bar counts it under."""
    def __init__(self, why: str, detail: str):
        super().__init__(f"{why}: {detail}")
        self.why = why


@dataclass
class Stats:
    """Request-level counters behind the gather_bar status line. gather_bar starts a fresh one; calls outside a bar add to whichever is current."""
    open: dict[int, float] = field(default_factory=dict)  # request id -> start time of its current http attempt
    sleeping: int = 0                                      # requests waiting out a backoff
    errors: Counter = field(default_factory=Counter)       # cause -> failed attempts
    finish: Counter = field(default_factory=Counter)       # finish_reason -> successful responses
    seen: dict[str, str] = field(default_factory=dict)     # cause -> first detail, printed once above the bar
    ok: int = 0                                            # successful responses
    latency: float = 0.0                                   # seconds, summed over successful attempts
    toks: int = 0                                          # completion tokens billed
    cost: float = 0.0                                      # dollars, from usage.cost

_stats = Stats()


def _verdict(status: int, body: dict, text: str) -> tuple[str | None, str]:
    """(None, "") for a usable response, else the (cause, detail) to retry on. 401/402 raise RuntimeError (no key, out of credits: stop everything); any other 4xx raises RequestFailed (retrying will not help)."""
    err, meta = body.get("error") or {}, (body.get("error") or {}).get("metadata") or {}
    detail = " | ".join(str(x) for x in (err.get("message"), meta.get("provider_name"), meta.get("raw"), meta.get("provider_error_code"), meta.get("limit_source")) if x) if err else text[:300]
    if status == 200 and body.get("choices"):
        choice = body["choices"][0]
        return (None, "") if choice.get("finish_reason") != "error" else ("finish error", json.dumps(choice)[:300])  # the provider aborted mid-stream; its usage counts are wrong too
    if status == 200:
        return f"envelope {err.get('code')}", detail  # a 200 whose body is an error: upstream 429s and provider failures come back this way
    if status in (401, 402):
        raise RuntimeError(f"{status}: {detail}")
    if status in (408, 429) or status >= 500:
        return str(status), detail
    raise RequestFailed(str(status), detail)


def _key() -> str:
    """The OpenRouter key, or a RuntimeError that says which .env was loaded. Importing mechtools loads the first .env found walking up from the cwd, so run from the project directory."""
    if "OPENROUTER_API_KEY" not in os.environ:
        found = find_dotenv(usecwd=True)
        raise RuntimeError(f"OPENROUTER_API_KEY is not set: {f'{found} was loaded when mechtools was imported and does not set it' if found else f'no .env found walking up from {os.getcwd()}'}. Put it in the project's .env and run from the project directory.")
    return os.environ["OPENROUTER_API_KEY"]


async def _post(client: httpx.AsyncClient, path: str, payload: dict, attempts: int, timeout: float) -> dict:
    """POST payload to path and return the body. Retries with doubling backoff (1, 3, 7, ... s) on network errors, timeouts, 408/429/5xx, a 200 with no choices, and finish_reason "error"; the first failure of each kind is printed."""
    s, rid, headers = _stats, id(payload), {"Authorization": f"Bearer {_key()}"}
    for attempt in range(attempts):
        if attempt:
            s.sleeping += 1
            await asyncio.sleep(2 ** attempt - 1)
            s.sleeping -= 1
        s.open[rid] = start = time.monotonic()
        try:
            r = await client.post(OPENROUTER_URL + path, json=payload, headers=headers, timeout=timeout)
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            why, detail = _verdict(r.status_code, body, r.text)
        except httpx.HTTPError as e:
            why, detail = "timeout" if isinstance(e, httpx.TimeoutException) else type(e).__name__, f"{type(e).__name__} {e}".strip()  # httpx timeouts stringify to nothing
        finally:
            s.open.pop(rid)
        if why is None:
            usage = body["usage"]
            s.ok += 1; s.latency += time.monotonic() - start
            s.toks += usage["completion_tokens"]; s.cost += usage["cost"]
            s.finish[body["choices"][0].get("finish_reason")] += 1
            return body
        s.errors[why] += 1
        if why not in s.seen:
            s.seen[why] = detail
            tqdm.write(f"  {yellow}{why}: {detail[:240]}{endc}")
    raise RequestFailed(why, detail)


async def _request(path: str, payload: dict, attempts: int, timeout: float, client: httpx.AsyncClient | None) -> dict:
    async with (contextlib.nullcontext(client) if client else httpx.AsyncClient()) as c:
        return await _post(c, path, payload, attempts, timeout)


def _usd(x: float) -> str:
    return f"${x:.2f}" if x >= 1 else f"${x:.4f}"


def _pin(provider: str | dict) -> dict:
    return {"only": [provider], "allow_fallbacks": False} if isinstance(provider, str) else provider


async def chat(conv: str | list[dict], model: str, provider: str | dict | None = None, reasoning: bool | str | dict | None = None, max_tokens: int = 8192, temperature: float = 1.0,
               attempts: int = 6, timeout: float = 600, client: httpx.AsyncClient | None = None, **body) -> dict:
    """One /chat/completions call: the response body as OpenRouter sent it (store that; flat pulls the common fields). A str conv is a single user turn. provider pins one endpoint by slug (a dict passes through as OpenRouter's provider object).
    reasoning True/False sets enabled, a str sets effort, a dict passes through. body passes through: top_p, top_k, seed, stop, logprobs, response_format, ..."""
    payload = {"model": model, "messages": [{"role": "user", "content": conv}] if isinstance(conv, str) else conv, "max_tokens": max_tokens, "temperature": temperature, "transforms": [], "usage": {"include": True}, **body}
    if provider: payload["provider"] = _pin(provider)
    if reasoning is not None: payload["reasoning"] = {"enabled": reasoning} if isinstance(reasoning, bool) else {"effort": reasoning} if isinstance(reasoning, str) else reasoning
    return await _request("/chat/completions", payload, attempts, timeout, client)


async def complete(prompt: str, model: str, provider: str | dict | None = None, max_tokens: int = 8192, temperature: float = 1.0, stop: str | list[str] | None = None,
                   attempts: int = 6, timeout: float = 600, client: httpx.AsyncClient | None = None, **body) -> dict:
    """One raw /completions call on a self-rendered prompt string, returning the response body, e.g. a chat template ending inside an open think block. transforms=[] keeps OpenRouter from compressing the prompt; whether the provider passes it through verbatim shows in prompt_tokens."""
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "transforms": [], "usage": {"include": True}, **body}
    if provider: payload["provider"] = _pin(provider)
    if stop: payload["stop"] = [stop] if isinstance(stop, str) else stop
    return await _request("/completions", payload, attempts, timeout, client)


def flat(body: dict) -> dict:
    """The common fields of either endpoint's body: text (message content, or the raw completion), reasoning, finish_reason, provider, model, prompt_tokens, completion_tokens, reasoning_tokens (wrong on many providers; count from text), cost (dollars). Store the body itself; it has more (reasoning_details with signatures, native_finish_reason, refusal, cache and cost breakdowns, the generation id)."""
    choice, usage = body["choices"][0], body["usage"]
    msg = choice.get("message")  # chat bodies; a raw completion carries text and reasoning on the choice itself
    return {"text": msg["content"] if msg else choice["text"], "reasoning": (msg or choice).get("reasoning"), "finish_reason": choice.get("finish_reason"), "provider": body["provider"], "model": body["model"],
            "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"], "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"), "cost": usage["cost"]}


async def gather_bar(coros: list, concurrency: int = 32, desc: str = "", swallow: tuple[type[Exception], ...] = (RequestFailed,)) -> list:
    """Awaits coros at most concurrency at a time under a progress bar; results in order. A coro that raises one of swallow yields None, counted by cause with the first of each kind printed above the bar; any other exception cancels the rest and propagates.
    The status line: coroutines ok/fail, dollars so far, open requests and the age of the oldest (slow models show here, not as errors), mean seconds per successful request, requests sleeping in backoff, finish reasons other than stop (length = truncated), failed attempts by cause (429, 503, timeout, envelope 429, finish error, ...). tqdm clips it to the terminal width; the summary printed at the end has everything."""
    global _stats
    _stats = s = Stats()
    sem, fails, n_ok, t0 = asyncio.Semaphore(concurrency), Counter(), 0, time.monotonic()
    bar = tqdm(total=len(coros), desc=f"{cyan}{desc}{endc}", bar_format="{desc} {bar:15} {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {unit}", ascii=" >=", dynamic_ncols=True)

    def status() -> str:
        parts = [f"{green}ok {n_ok}{endc}" + (f" {red}fail {sum(fails.values())}{endc}" if fails else ""), f"{cyan}{_usd(s.cost)}{endc}"]
        if s.open: parts.append(f"open {len(s.open)} oldest {time.monotonic() - min(s.open.values()):.0f}s")
        if s.ok: parts.append(f"{s.latency / s.ok:.0f}s/req")
        if s.sleeping: parts.append(f"{yellow}backoff {s.sleeping}{endc}")
        if odd := {k: v for k, v in s.finish.items() if k != "stop"}: parts.append(yellow + " ".join(f"{k} {v}" for k, v in odd.items()) + endc)
        if s.errors: parts.append(f"{yellow}errs " + " ".join(f"{k}×{v}" for k, v in s.errors.most_common()) + endc)
        return " | ".join(parts)

    async def run(coro):
        nonlocal n_ok
        async with sem:
            try:
                r = await coro
                n_ok += 1
            except swallow as e:
                why = e.why if isinstance(e, RequestFailed) else type(e).__name__
                fails[why] += 1
                if fails[why] == 1: bar.write(f"  {red}failed ({why}): {str(e)[:240]}{endc}")
                r = None
            bar.unit = status()
            bar.update()
            return r

    async def tick():
        while True:
            await asyncio.sleep(1)
            bar.unit = status()
            bar.refresh()

    ticker = asyncio.create_task(tick())
    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(run(c)) for c in coros]
    finally:
        ticker.cancel()
        bar.close()
        fail_str = f", {red}{sum(fails.values())} failed: " + ", ".join(f"{k}×{v}" for k, v in fails.most_common()) + endc if fails else ""
        err_str = f" | {yellow}errors " + ", ".join(f"{k}×{v}" for k, v in s.errors.most_common()) + endc if s.errors else ""
        print(f"  {desc or 'batch'}: {green}{n_ok} ok{endc}{fail_str} | finish " + ", ".join(f"{k} {v}" for k, v in s.finish.most_common()) + err_str
              + f" | {s.toks:,} completion toks {cyan}{_usd(s.cost)}{endc} | {s.latency / max(s.ok, 1):.1f}s/req, {time.monotonic() - t0:.0f}s total")
    return [task.result() for task in tasks]


async def chat_batch(convs: list, model: str, concurrency: int = 32, desc: str = "", **kw) -> list[dict | None]:
    """gather_bar over chat with one shared http client; [conv] * n samples one prompt n times. kw goes to chat."""
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)) as client:
        return await gather_bar([chat(c, model, client=client, **kw) for c in convs], concurrency, desc)


async def complete_batch(prompts: list[str], model: str, concurrency: int = 32, desc: str = "", **kw) -> list[dict | None]:
    """gather_bar over complete with one shared http client. kw goes to complete."""
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)) as client:
        return await gather_bar([complete(p, model, client=client, **kw) for p in prompts], concurrency, desc)


def endpoints(model: str) -> list[dict]:
    """The endpoints serving model, each with the slug to pin under "provider" plus the API's fields: tag, quantization, context_length, max_completion_tokens, pricing, supported_parameters, ..."""
    r = httpx.get(f"{OPENROUTER_URL}/models/{model}/endpoints", headers={"Authorization": f"Bearer {_key()}"}, timeout=30)
    r.raise_for_status()
    return [{"provider": e["tag"].split("/")[0], **e} for e in r.json()["data"]["endpoints"]]
