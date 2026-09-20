# OpenRouter provider findings for CoT resampling (from the weirdchat project)

Compiled 2026-09-13 from the weirdchat repo (`utils.py`, the three `*_resample.py` configs, `run.py`), the project memory files, and the Claude Code session transcripts of the probe sessions (2026-09-02/03 for DeepSeek V4 Flash and Qwen3.6-27B, 2026-09-03 for Inkling) plus a sweep of every other session in the project. Everything here was measured on those dates. OpenRouter's provider roster, templates, and defaults change; re-run the cheap checks in section 12 before trusting a provider in a new run.

Models: `deepseek/deepseek-v4-flash`, `qwen/qwen3.6-27b`, `thinkingmachines/inkling` (resampled), and `qwen/qwen3.6-35b-a3b`, `qwen/qwen3-8b`, `google/gemma-4-31b-it`, `nvidia/nemotron-3-ultra-550b-a55b` (replay only, lighter findings).

## 1. Summary

- Chat-message `<think>` prefill (the paulcbogdan/rollouts approach: trailing assistant message with `"<think>\n" + partial`) does not continue the CoT on any of the 24 OpenRouter endpoints tested for DeepSeek V4 Flash and Qwen3.6-27B. Both official templates wrap assistant content in a closed think block, so every provider renders a finished turn and the model starts fresh. Four DeepSeek providers render a prompt ending in EOS and the model emits multilingual garbage.
- Raw `/api/v1/completions` with a self-rendered prompt string ending inside an open think block works on a subset of providers, verified by exact `usage.prompt_tokens` match to the local tokenizer at several prefix lengths (up to 2029 tokens) and by first-token continuation. Passing: DeepSeek V4 Flash on CoreWeave, DeepInfra, Mancer, Parasail; Qwen3.6-27B on Chutes, CoreWeave, Phala; Inkling on Together only.
- Everything else either wraps the raw prompt in a chat template (constant +4, +10, +14, +50 token offsets, or ~+80 with an injected system prompt) or rejects the endpoint. Alibaba rejects raw completions for both DeepSeek and Qwen, so the Alibaba-pinned replay runs cannot be resampled on the provider that produced them.
- The passing endpoints return the generation split at `</think>` into `choices[0].reasoning` and `choices[0].text` with the tag dropped. Together (Inkling) returns reasoning and response merged with all special tokens stripped, so Inkling rollouts need a `stop` string and two calls.
- `usage.completion_tokens` matches the local tokenizer plus a small constant on every provider. `usage.completion_tokens_details.reasoning_tokens` is wrong on about half of DeepSeek's providers and reads 0 on Chutes/Phala/Together. Count reasoning length from text.
- Provider sampling defaults differ: CoreWeave applies Qwen3.6-27B's `generation_config` (top_k 20, top_p 0.95) when the request omits them; Chutes and Phala do not; DeepSeek has no truncation defaults; Together shows no top_p 0.95 truncation on Inkling. Set top_p/top_k explicitly if provider independence matters.
- Cross-provider resample curves agree within Wilson intervals at S=50 (DeepInfra vs Parasail, Chutes vs CoreWeave). Provider effects on base rates are plausible but not demonstrated at that resolution.
- Rate limiting is the dominant operational cost: DeepInfra 429s in bursts independent of request rate, Together 503/429 above ~12 concurrent two-stage rollouts, BaseTen and Parasail shared-pool 429s. OpenRouter also returns HTTP 200 bodies that are error envelopes with `choices: null`.

## 2. Request shapes that were used

Chat replay (`utils.sample_once`, via `AsyncOpenAI(base_url="https://openrouter.ai/api/v1", timeout=300, max_retries=3)`):

```python
client.chat.completions.create(model=..., messages=[...], temperature=1.0, max_tokens=1024 or 8192 (Inkling 32768),
    extra_body={"reasoning": {"enabled": bool}, "provider": {"order": [provider], "allow_fallbacks": False}})
# trace: reply.choices[0].message.model_extra["reasoning"]; served provider: reply.model_extra["provider"]
# reasoning tokens: reply.usage.completion_tokens_details.reasoning_tokens (unreliable, see section 6)
```

Raw completions for resampling (`utils._complete`, plain httpx):

```python
payload = {"model": model, "prompt": prompt_str, "max_tokens": 8192, "temperature": 1.0, "transforms": [],
           "provider": {"only": [provider], "allow_fallbacks": False}}
if stop: payload["stop"] = [stop]
r = await client.post("https://openrouter.ai/api/v1/completions", json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=300)
```

`transforms: []` disables OpenRouter's middle-out prompt compression. `provider.only` and `provider.order` + `allow_fallbacks: false` both pin; `only` was used for raw completions, `order` for chat. The endpoints list for a model comes from `GET /api/v1/models/{model}/endpoints` (`data.endpoints[].tag`, `.quantization`, `.supported_parameters`, prices); the provider slug is `tag.split("/")[0]`.

## 3. Chat-message think prefill: fails everywhere (tested 2026-09-02)

Method. For each (model, endpoint), to `/chat/completions` with `reasoning: {enabled: true}`, `include_reasoning: true`:
- A0: user turn only, `max_tokens 1`. A1 / A2: user + assistant `"<think>\n" + X` with a short and a long X. A1_off: same as A1 with reasoning off. All read `usage.prompt_tokens`.
- Locally render the same messages under every plausible template hypothesis (official template with the assistant content as a finished turn with/without EOS; a raw continuation; content placed in `reasoning_content`) with the official Qwen3.6 Jinja template and DeepSeek's shipped `encoding/encoding_dsv4.py`, count tokens with the HF tokenizer (`add_special_tokens=False`), and match the reported count to a hypothesis. Providers that inject a system prompt are matched on the delta from their own A0 count.
- B: 8 samples at `max_tokens 12` with a prefix ending on a clean token boundary (`"... 340 + 51 ="`) and the expected continuation a single token (`" 391"` is one DeepSeek token; the first attempt ended the prefix with `"= 3"` and was unfair to DeepSeek). Score: starts with the expected token vs matches a restart regex (`Okay|The user|We need|Here|Let me|Alright|...`).

Results.
- DeepSeek, 17 endpoints: all 17 report a closed-turn rendering; 0 of 136 B samples continued. Azure, Baidu, Cloudflare, Parasail, Phala, StreamLake, NextBit render the closed turn without EOS (count 51 with reasoning on). CoreWeave, DeepInfra, DigitalOcean, Venice render the official `std_on` which ends in `<｜end▁of▁sentence｜>`, so the model samples past EOS: `' Jeg Ger dig svaret: 17 * 23 = 391'`, `' Các kết quả khác: 17*23 ='`, `' Wombat\n\nWait, that's 391.'`. Alibaba and AtlasCloud sit +3 over the official rendering. GMICloud, Novita, SiliconFlow inject a ~80-token system prompt when reasoning is enabled (A1 134/135/136 vs 55). SiliconFlow's count alternated 132/136 across identical requests, Mancer 53/55, NextBit 51/56: heterogeneous backends behind one slug.
- Qwen3.6-27B, 7 endpoints: all 7 closed-turn; 1 of 59 B samples began with `391` (Phala), 55 of 59 restarted (`"Here's a thinking process:\n\n1.  **Analyz"`, `'The user wants to know the product of 17'`). CoreWeave, DeepInfra, SiliconFlow, Venice render the exact official template (81 tokens); Chutes and Phala omit the empty `<think>\n\n</think>\n\n` block (77); Alibaba is +2 (83). Reasoning-off adds the 2 tokens the template predicts on every endpoint. Alibaba errored on 5 of 8 B requests.
- Not tested but structurally excluded: putting the partial CoT in the assistant message's `reasoning` field (both templates render it closed), and DeepSeek's first-party prefix mode (the encoder always appends `</think>` after `reasoning_content`).

Confidence that no chat-message encoding continues an open CoT on these providers: ~97%. The token counts are unambiguous and the behavioral test agrees on every endpoint.

## 4. Raw /completions passthrough (tested 2026-09-02 for DeepSeek/Qwen, 2026-09-03 for Inkling)

Method. Six self-rendered prompt strings per model: P0 = generation prompt ending inside the open think block; P1 / P2 = P0 + short / long partial CoT; PW = P0 + text with odd whitespace (tabs, four newlines); PF = a finished CoT + closing tag; PE = the reasoning-off prompt. One call each at `max_tokens 1` (plus P1 with `reasoning: {enabled: true/false}` to see whether the flag changes the count), 8 calls of P1 at `max_tokens 12` for first-token continuation, 3 each of PF and PE at `max_tokens 20`. Compare `usage.prompt_tokens` to the local count of the exact string. A pass is exact match on all six strings plus " 391" continuations. Later confirmed at full length (record 768's 2029-token prefix, exact on Chutes/CoreWeave/Phala) and per rollout inside the resampler (`prompt_tokens == n_prompt + t` on all 1800 DeepInfra rollouts and 8669 of 8670 Together rollouts; the one exception had `finish_reason: "error"`).

### DeepSeek V4 Flash (local counts P0 12, P1 49, P2 192, PW 26, PF 196, PE 12)

| provider | quant | prompt_tokens vs local | verdict | notes |
|---|---|---|---|---|
| CoreWeave | fp8 | exact on all six | pass | logprobs available; `reasoning_tokens` correct; $0.14/$0.28 per M |
| DeepInfra | fp8 | exact | pass | cheapest ($0.09/$0.18); 429 bursts (section 10) |
| Mancer 2 | fp8 | exact | pass | $0.185/$0.50; logprobs |
| Parasail | fp8 | exact | pass | $0.14/$0.28; logprobs |
| Azure, Baidu, Cloudflare, DigitalOcean, StreamLake, Venice | fp8 / unlisted | +4 constant | fail, chat-wrapped | prompt becomes a user turn; model answers from scratch (`'The product of 17 and 23 is 391.'`); Baidu/StreamLake/Venice return empty `text` with output in `reasoning` |
| SiliconFlow | fp8 | +83 with reasoning, +4 with reasoning off | fail | injected system prompt plus wrap |
| Alibaba, NextBit, Novita, Phala | fp8 / unlisted | `Provider returned error` | rejected | Phala passes for Qwen but not DeepSeek |
| AtlasCloud, GMICloud | fp4 / fp8 | `Provider returned an empty response` | rejected | |

### Qwen3.6-27B (local counts P0 20, P1 68, P2 242, PW 38, PF 250, PE 22)

| provider | quant | prompt_tokens vs local | verdict | notes |
|---|---|---|---|---|
| Chutes | fp8 | exact | pass | cheapest ($0.30/$2.00); `reasoning_tokens` reads 0; seed ignored; no default top_p/top_k |
| CoreWeave | fp8 | exact | pass | $0.60/$3.60; logprobs; `reasoning_tokens` correct; applies top_k 20 / top_p 0.95 by default (section 7) |
| Phala | unlisted | exact | pass | $0.32/$2.70; `reasoning_tokens` reads 0; no default truncation |
| DeepInfra, SiliconFlow, Venice | fp8 | +10 (+12 with reasoning off) | fail, chat-wrapped | all fields empty at `max_tokens 12` |
| Alibaba | unlisted | `Provider returned error` | rejected | the provider every Qwen replay run pinned |

### Inkling (local counts with the effort-0.9 header and `<|content_thinking|>` appended: 66 for the probe record)

| provider | quant | prompt_tokens vs local | verdict | notes |
|---|---|---|---|---|
| Together | unknown | exact at t=0 (66) and t=100 (166); 8669/8670 rollouts exact | pass, two-stage only | returns reasoning+response merged, special tokens stripped, `reasoning` field absent, `reasoning_tokens` 0; `reasoning: {enabled}` and `include_reasoning` on the raw endpoint change nothing; `stop` honored; $1.00/$4.05 |
| DeepInfra | fp8 | +14 constant | fail, chat-wrapped | at t=100 it skipped thinking and answered directly |
| BaseTen | fp8 (listed twice) | +50 constant | fail, wrapped + system prompt | 429 `upstream_provider_shared_pool` on most probe requests |

### Other models (chat endpoint only, no raw-completions probe was run)

- Qwen3.6-35B-A3B: served by Parasail (1666 records) and eight small providers after unpinning (Darkbloom, AkashML, Venice, DeepInfra, CoreWeave, SiliconFlow, AtlasCloud, Io Net). Every record's `prompt_tokens` equals the local no-system-prompt rendering (zero offset). Parasail 429'd (`limit_source: upstream_provider_shared_pool`) at 452/576 reasoning-on samples.
- Qwen3-8B: one endpoint (Alibaba, max completion 8192, quant unknown, $0.117/$0.455). Pinning is moot; raw passthrough untested and Alibaba rejects it for the other two models, so expect no.
- Gemma 4 31B: 16 endpoints list reasoning; DeepInfra returned a 987-char trace against 256 billed reasoning tokens (3.9 chars/token, taken as verbatim not summarized). Replays served mostly by DeepInfra (919/1152 on).
- Nemotron 3 Ultra: 4 endpoints (DeepInfra, BaseTen x2, Venice); BaseTen served 1149/1152 replay records; reasoning traces are short on easy prompts (19 tokens on `17 * 23`), which is model behavior, not truncation.
- The closed labs (Gemini, OpenAI, xAI, Anthropic >= 4.6) return summarized or encrypted reasoning only; no OpenRouter routing recovers raw CoT for them (checked 2026-08-30 in another project). Open-weight reasoners return raw traces as `reasoning_details[].type == "reasoning.text"`.

## 5. Return path on the passing endpoints

DeepSeek and Qwen (Chutes, CoreWeave, Phala, DeepInfra, Mancer, Parasail): the raw endpoint splits the generation at `</think>` into `choices[0].reasoning` (before) and `choices[0].text` (after), plus `reasoning_details = [{"type": "reasoning.text", ...}]`. The tag is in neither field; on Qwen the `\n\n` after it is dropped too. Reconstruct as `reasoning + "</think>" + text` (Qwen: `+ "\n\n"`). Accounting from 21 long samples, all `finish_reason: stop`: `completion_tokens = tokens(reasoning) + tokens(text) + 2` on DeepSeek (closing tag + EOS) and `+ 3` on Qwen (closing tag, the stripped `\n\n`, EOS). `reasoning_tokens` is 0 on Chutes and Phala with populated reasoning.

Together / Inkling: one call returns `text = reasoning + response` with no boundary (`' 391.The answer is **391**'`), `completion_tokens` = local text tokens + 6 (the dropped special tokens). Verified fix:
- Stage 1: prompt = prefix, `stop: ["<|end_message|>"]`. Returns only the reasoning continuation (`completion_tokens` = local + 1 for the stop token, excluded from text).
- Stage 2: prompt = prefix + continuation + `<|end_message|><|message_model|><|content_text|>`, same stop. `prompt_tokens` exact (357 = 66 + 288 + 3), returns the visible response only.
- Caveat: stage 2 forces a text message where the model could open a second thinking message. The three unstopped 400-token samples all showed one thinking message then the answer, but that is a small sample.
- Together sets `finish_reason: "error"` when a generation dies mid-stream, and the usage on such a response is wrong (one rollout reported 388 prompt tokens against 308 for its 29 siblings). 4 of 8670 rollouts. The harness raises on it.
- Reasoning-off Inkling records render as `Thinking effort level: 0` and the text block opened directly (`<|content_text|>`); Together records then have `completion_tokens = tokens(response) + 3`.

## 6. Token accounting in the chat replay records (what the resampler asserts against)

Per-record `completion_tokens - (tokens(reasoning) + tokens(response))`, all reasoning-on records:
- DeepSeek, 17 providers: 2 on nearly every record (tag + EOS); GMICloud and NextBit mostly 1; SiliconFlow mixed 1/2; Azure has a 7% tail of large gaps (truncated traces). Overall 93-100% within {1, 2} per provider.
- Qwen3.6-27B on Alibaba: 2, with `reasoning_tokens` off by exactly +1 on 776 of 2880 records (a trailing newline counted with the tag) and one record off by 2578.
- Inkling: 6 on Together (1026/1034), 7 on BaseTen (2808/2833) and DeepInfra (1161/1173).

`reasoning_tokens` vs local `tokens(reasoning)` on DeepSeek reasoning-on records in `dv4f_full_elo` (n, fraction equal): GMICloud 3361 / 99.9%; StreamLake 2830 / 99.7%; NextBit 1149 / 99.7%; AtlasCloud 1139 / 99.9%; Novita 1141 / 99.3%; Phala 572 / 99.1%; Baidu 88 / 100%; Alibaba 1247 / 96.9%; SiliconFlow 1173 / 95.8%; DigitalOcean 2910 / 16.8%; DeepInfra 3181 / 0.1%; Parasail 1131 / 0.1%; CoreWeave 1123 / 0.1%; Venice 1161 / 0.0%; Mancer 2 621 / 0.0%; Cloudflare 119 / 0.0%; Azure 332 / 0.0%. The bad providers count 30-250 tokens of the response as reasoning (Venice by hundreds on long traces); the texts themselves are intact everywhere. Any "mean reasoning tokens" statistic read from that field is inflated wherever those providers served.

Prompt tokens: every Qwen record (15072 Alibaba + 1728 for the 35B) equals the local rendering with no system prompt exactly, so no Qwen provider injects one on the chat endpoint. DeepSeek records equal the local `encode_messages(..., "thinking")` rendering (record 128: 90 = 90). Inkling records report local-minus-1 on Together/BaseTen and local-minus-2 on DeepInfra, where "local" includes the `<|content_thinking|>` the model would generate itself.

The resampler therefore asserts `n_prompt_local == record.prompt_tokens + prompt_extra` and `record.completion_tokens - local(reasoning + response) in completion_extra`, with `prompt_extra` 0 (DeepSeek, Qwen), 1 (Inkling on Together/BaseTen), 2 (Inkling on DeepInfra) and `completion_extra` (1, 2) / (5, 6, 7) / (3,) for the Inkling reasoning-off case. A larger gap means a truncated trace.

## 7. Prompt rendering per model

- Qwen3.6-27B: `tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)` ends with `<|im_start|>assistant\n<think>\n`. The FP8 checkpoint revisions the dataset used (27B `e89b16e`, 35B `95a723d`) have chat templates byte-identical to the bf16 repos. `generation_config.json`: top_k 20, top_p 0.95, temperature 1.0.
- DeepSeek V4 Flash: no Jinja template; the checkpoint ships `encoding/encoding_dsv4.py` (get it with `hf_hub_download("deepseek-ai/DeepSeek-V4-Flash", "encoding/encoding_dsv4.py")`, `sys.path.insert` its directory, `from encoding_dsv4 import encode_messages`). `encode_messages(msgs, "thinking")` ends with `<｜Assistant｜><think>`; `"chat"` mode ends with `<｜Assistant｜></think>`. `generation_config.json`: top_p 1.0, no top_k.
- Inkling: HF `thinkingmachines/Inkling` template (no BOS; `tok('hi')` adds nothing). It emits a `Thinking effort level: <x>` system message first; `reasoning_effort` maps `none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9, max 0.99`, default 0.9 when unset. `add_generation_prompt=True` ends at `<|message_model|>`; append `<|content_thinking|>` to open thinking or `<|content_text|>` for a reasoning-off response. No `generation_config.json`.
- Inkling effort ambiguity: the OpenRouter models API says `default_effort: "high"` for `reasoning: {enabled: true}`, the docs say enabled means "medium". 0.7 and 0.9 render to the same token count, and S=30 trace lengths cannot separate them (raw 0.2 / 0.7 / 0.9 / 0.99 = 203 / 478 / 471 / 543 mean tokens; chat enabled / medium / high = 528 / 502 / 519; Together replay records 506). Separating them would need a few hundred samples per arm. Reasoning-off replays rendered effort 0 (record prompt_tokens 63 = local effort-0 rendering).
- Byte-level BPE: a cut inside a multi-byte character decodes to a replacement char and does not retokenize to the same ids. Survey: 290 of 2304 Qwen records (1610 of 2.99M positions) have such positions; 1 of 1536 DeepSeek records. The resampler checks `tok(tok.decode(ids[:t])) == ids[:t]` per position and skips failures.
- Control-token strings in dataset prompts: 0 hits across 2661 prompts. 35 of 58735 replay outputs contain one, all DeepSeek claims-device-access responses emitting `<｜DSML｜tool_calls>` as text.

## 8. Provider sampling defaults (tested 2026-09-03)

Method: raw `/completions` prompt at a high-entropy position with thinking closed (`"Once upon a time, in a land far away, there lived a"`), `max_tokens 1`, temperature 1, pinned provider, 500 single-token draws (1500 for the DeepSeek check) under (a) request omitting top_p/top_k, (b) explicit `top_p: 1.0, top_k: 0`, (c) explicit `top_p: 1.0` only. Compare distinct-token counts, tail tokens, and total variation distance (noise floor at n=500 ~0.10-0.15 for a broad distribution). Logprobs were not usable: Qwen `top_logprobs` returned 0 tokens on CoreWeave; DeepSeek returned 20 tokens for a different position than the sampled one.

| model, provider | omitted params | explicit params | verdict |
|---|---|---|---|
| Qwen, CoreWeave | 4 distinct tokens in 500 draws, three replications | 9 to 13 distinct | applies the checkpoint's top_k 20 / top_p 0.95 |
| Qwen, Chutes | 13 | 13 | no truncation |
| Qwen, Phala | 16 | 8 | no truncation (tail present both ways) |
| DeepSeek, CoreWeave | 122 distinct in 1500 draws, tail mass past rank 40 = 0.066 | 112 distinct, 0.063; TV 0.097 | no truncation |
| Inkling, Together | 7 distinct in 400 draws at a paragraph boundary; singletons at p~0.0025 past 0.96 cumulative mass | not run | no top_p 0.95; top_k not testable with 7 distinct |

Alibaba's defaults are unknown (raw endpoint rejected, not probed via chat). The chat replay runs omitted top_p/top_k, so unpinned Qwen samples served by CoreWeave came from a truncated distribution.

Seed: `seed` gives identical 40-token outputs on DeepSeek via DeepInfra and CoreWeave and different outputs for a different seed. Qwen via CoreWeave and Chutes ignore the seed.

## 9. Cross-provider agreement and quantization

S=50 at three positions, same record, two passing providers each, production rollout path (judge verdict per rollout):

| record, position | provider A | provider B | replay rate on the prompt |
|---|---|---|---|
| DeepSeek 128 t=0 | DeepInfra 0.18 [0.10, 0.31] | Parasail 0.26 [0.16, 0.40] | 14/50 = 0.28 (13 providers mixed) |
| DeepSeek 128 t=30 | 0.48 [0.35, 0.61] | 0.52 [0.39, 0.65] | |
| DeepSeek 128 t=46 | 0.80 [0.67, 0.89] | 0.80 [0.67, 0.89] | S=20 had read 20/20 here |
| Qwen 768 t=0 | Chutes 0.24 [0.14, 0.37] | CoreWeave 0.16 [0.08, 0.29] | 2/32 = 0.06 on Alibaba |
| Qwen 768 t=960 | 0.68 | 0.64 | |
| Qwen 768 t=1920 | 0.92 [0.81, 0.97] | 0.82 [0.69, 0.90] | |

Every pair overlaps; this resolution only detects gaps above ~0.2. Qwen's o_0 on the fp8 hosts (0.16-0.24) vs the Alibaba replay rate (2/32) is suggestive of a provider effect on the base rate (quantization or Alibaba's sampling defaults), not conclusive. On Inkling the three chat providers agree on reasoning-on match rates (BaseTen 0.008 n=2833, DeepInfra 0.007 n=1173, Together 0.005 n=1034), and the record-7979 resample through Together gave 199/5944 = 3.4% at t=0 vs 9/240 = 3.8% in the mixed-provider replay.

Base rollouts and resamples should come from the same endpoint. Where that was not possible (record 128 base from Baidu, resampled on DeepInfra; Inkling 7979 base from DeepInfra, resampled on Together) the curve measures P(match | prefix) under the resampling provider's deployment. Dataset-side checkpoints were FP8 (Qwen), mixed FP4/FP8 with fp8 KV cache (DeepSeek), NVFP4 (Gemma, Nemotron, Inkling) on SGLang; OpenRouter serving precision is whatever the endpoint lists, "unknown" for Phala, Alibaba (Qwen), Together, Venice (DeepSeek).

## 10. Operational findings (rate limits, errors, concurrency)

- DeepInfra (DeepSeek raw completions): 429 `temporarily rate-limited upstream` in bursts, independent of request rate. 1800 rollouts at concurrency 32 with 6 attempts and `sleep(2**attempt - 1)`: 132 failures on the main pass, then fill passes of 29/88 etc; a loop of up to 5 passes with 45 s pauses reached 1800. About a third of first-pass requests hit the throttle. Without retries at concurrency 64 the first pass died within a minute (1488 failures).
- Together (Inkling): at concurrency 32 the run produced 1136/8670 rollouts in 10 min (~5.4 s per two-stage rollout, ~17 s latency) then hit 503 `Service unavailable` and 429 `temporarily rate-limited upstream`. Recommendation adopted: stop, wait ~10 min, restart at concurrency 12. 2068 rollouts in 14:40 on a later pass.
- BaseTen: 429 `limit_source: upstream_provider_shared_pool` on 6 of 8 retried probe requests; also the suspected source of HTTP 200 error envelopes.
- Parasail (Qwen 35B chat): shared-pool 429 mid-run; unpinning let OpenRouter route the remainder to eight other providers.
- Alibaba (chat): sustained concurrency 196 on variant runs (256 reasoning-off samples in 4:25, reasoning-on at 13.7 tasks/s) until a 200-with-no-choices body crashed the process. Two processes at 196 on the same provider.
- HTTP 200 error envelopes: OpenRouter sometimes returns a 200 whose JSON is `{"error": ...}`; the OpenAI SDK parses it into a `ChatCompletion` with `choices=None` and `choices[0]` raises `TypeError`. Seen at concurrency 128 (Inkling, BaseTen) and 196 (Alibaba). Guard by checking `body["choices"]` (the raw client does `body["choices"][0]` inside the retry loop, so it retries).
- Chat retries: SDK `max_retries=3` plus 3 attempts with `sleep(2**attempt)`; raw completions: 6 attempts. Rate-limit errors from the SDK show as `RateLimitError`; a `"empty response"` failure is not throttling.
- Empty responses on Inkling reasoning-on were cap exhaustion: `max_tokens` covers reasoning plus response, one prompt's traces had median 7381 / max 8091 tokens against 8192. Inkling runs use 32768. Store `finish_reason` so this is diagnosable (the replay records did not; the resampler does).
- Echo/prompt logprobs are unavailable on the raw endpoint: `{"echo": true, "logprobs": 1}` returns 400 (CoreWeave, Parasail, Mancer, Chutes) or 422 (DeepInfra); `{"logprobs": true}` is accepted and ignored except CoreWeave/DeepSeek, which returns completion-token logprobs only. So prompt-token identity cannot be confirmed by logprobs on OpenRouter; only via `prompt_tokens` counts plus behavior. A self-hosted vLLM would allow it.
- Throughput seen on the passing endpoints: DeepInfra ~1.4 s per DeepSeek rollout at concurrency 24; Chutes ~2.8 s per Qwen rollout; Together ~3-5 s per two-stage Inkling rollout at low concurrency. Cost: 1800 DeepSeek rollouts (median 208 completion tokens) $0.10 subject-side; an Inkling stride-1 S=30 run on a ~300-token trace ~$15-25; Qwen3.6-27B is 4-20x pricier per token than DeepSeek V4 Flash. The judge (gemma-4-31b-it, `reasoning_effort="medium"`, trace billed and dropped) costs roughly as much as the subject on cheap models; ~2950 input tokens per judge call.

## 11. Resampler design points worth carrying over

- One judged base record in, `rollouts.jsonl` out (one line per rollout: `t, i, reasoning_cont, response, finish_reason, provider, prompt_tokens, completion_tokens, category, judge_match, judge_explanation`), plus `scores.json` with per-position `p_match` and Wilson interval. Reruns fill the per-position deficit against `S`, so a throttled pass is just rerun. Flush per rollout.
- Position grid: every `stride`-th token plus 0 and the end; positions whose decoded prefix does not retokenize identically are skipped with a note.
- Per-rollout invariant: `body.usage.prompt_tokens == n_prompt_local + t`. This catches a provider wrapping or re-tokenizing the prefix and a mid-stream `finish_reason: "error"`.
- Three outcome categories: match, nomatch, other (empty response, or trace hit `max_tokens` before the response). Truncated traces are still judged in the chat replays; in the resampler a stage-1 `length` finish skips stage 2 and records `other`.
- `cut="response"` mode: cut a reasoning-off record's visible output and judge prefix + continuation whole; needs the reasoning-off rendering (Inkling: effort 0 + `<|content_text|>`, `max_tokens 1024`).
- S=20 gives intervals ~0.4 wide mid-range; S=50 narrows to ~0.25. A 20/20 at S=20 read 40/50 at S=50.

## 12. Test recipes for a new model or provider

1. List endpoints: `GET /api/v1/models/{model}/endpoints`; note `quantization`, `supported_parameters` (`stop`, `top_k`, `seed`, `logprobs`), prices, and whether a slug appears twice.
2. Render the prompt locally with the official template so it ends inside an open think block; count tokens with `add_special_tokens=False`. Check whether the tokenizer adds BOS on its own.
3. For each provider: raw `/completions` at `max_tokens 1` with the bare generation prompt, with a ~100-token partial CoT, with a whitespace-heavy string, and with a finished CoT plus closing tag. Pass = `usage.prompt_tokens` exactly equals the local count on all of them. A constant offset means a chat wrapper; ~+80 means an injected system prompt; an error means the endpoint is not served raw. Also confirm at full trace length.
4. First-token check: 8 samples at `max_tokens 12` from a prefix ending on a token boundary whose expected next token is a single token; expect all 8 to continue rather than restart. Read both `text` and `reasoning` fields.
5. Return path: send one long completion; look for special tokens in the returned text (`[s for s in tok.all_special_tokens if s in text]`), check whether `reasoning` is populated, whether `reasoning_tokens` is 0, and reconcile `completion_tokens` with local token counts of the returned fields to learn the constant. If reasoning and text come merged, test `stop` at the end-of-thinking token and a two-stage rollout with exact `prompt_tokens` at stage 2.
6. Check `reasoning: {enabled}` / `include_reasoning` on the raw endpoint change nothing (they should not).
7. Sampling defaults: 500 single-token draws at a high-entropy position, request with and without explicit `top_p 1.0, top_k 0`; compare distinct counts and tail. Also test seed determinism if needed.
8. Cross-provider check: S=50 at 3 positions on two passing providers, expect Wilson overlap.
9. In production: assert `prompt_tokens == n_prompt + t` per rollout, treat `finish_reason == "error"` as a failure, retry on 429/503 with backoff, and keep concurrency near 12-32 for raw completions.
10. For chat-prefill claims on any new model: render the hypotheses locally and match `prompt_tokens`; do not trust the response text alone.

## 13. Credences (as stated in the sessions, with today's adjustment)

| claim | credence then | now (2026-09-13) |
|---|---|---|
| chat-message think prefill cannot continue a CoT on OpenRouter for these models | ~97% | same; structural |
| model receives exactly the rendered tokens on the passing endpoints (DeepSeek x4, Qwen x3) | ~96% after length checks and dataset scans | ~96% for the endpoints as measured; ~80% that each still passes unchanged today (providers re-deploy); the per-rollout assertion catches a change |
| same for Inkling on Together | ~85% | ~85%; the merged return path means fewer independent checks |
| returned text is a faithful split of what the model generated | ~90% (DeepSeek, Qwen), ~80% Inkling two-stage | same; based on 21 + a handful of long samples and 100% `stop` finishes over thousands of rollouts |
| sampling distribution matches the original rollouts' provider | DeepSeek ~80%; Qwen Chutes ~75%, CoreWeave ~20% without explicit params | same; the Alibaba vs fp8-host gap on Qwen (2/32 vs 0.16-0.24) is the open question |
| residual 4-5% on prompt fidelity | "provider reports a count of my string but feeds the model something else" and tokenizer revision drift | same |

## 14. Untested or open

- S=400 cross-provider check (proposed ~$0.40 DeepSeek, ~$25 Qwen); Inkling effort at a few hundred samples per arm (~$5).
- Alibaba's sampling defaults (raw endpoint rejected); whether a chat-endpoint probe at a high-entropy position could measure them.
- Chutes and CoreWeave at thousands of concurrent requests; DeepInfra throttled hard at 1800.
- Direct provider APIs (DeepInfra, Fireworks, Together first-party) for prompt logprobs / echo, which OpenRouter lacks.
- Whether Together's second stage ever suppresses a second thinking message that the model would have opened.
- The Baidu split in `dv4f_full_elo` (2256 reasoning-off vs 88 reasoning-on records) is unexplained; Baidu's counts were otherwise clean.
- Gemma and Nemotron raw passthrough, and their trace verbatim-ness beyond the chars-per-token ratio.
- Replay runs other than Qwen3.6-27B were unpinned (DeepSeek, Qwen 35B, Inkling, Gemma, Nemotron), while every DeepSeek and Qwen variant/ablation run was pinned to Alibaba; provider-mix effects on rates were flagged as plausible but never isolated.

## 15. Sources

- `~/wgmn/weirdchat/utils.py` (`sample_once`, `gather_bar`, `ResampleConfig`, `_complete`, `_rollout`, `_resample`), `deepseek_resample.py`, `resample_qwen.py`, `inkling_resample.py`, `run.py`.
- Memory files: `~/.claude/projects/-home-ek-wgmn-weirdchat/memory/openrouter-think-prefill-fails.md`, `inkling-resample-endpoint.md`; `~/.claude/projects/-home-ek-wgmn-odd-number-hacking/memory/reasoning-trace-exposure.md`, `pin-single-provider.md`.
- Session transcripts: `4b1057c0` (2026-09-02/03, DeepSeek/Qwen probes, sampling defaults, S=50), `858a29d7` (2026-09-03, Inkling), `acc1269c` (token audit), `75d3aa9c` (DeepInfra 429s), `76fed566` / `619e5fb6` (200 error envelopes), `f8ec3372` (Inkling cap), `83dd9db4` / `999e6b04` / `479fadd0` (Inkling provider agreement, finish_reason error, response mode), `81a2af6d` / `5e0e2c94` / `5fc300d7` (Qwen endpoints, zero offsets, Qwen3-8B), `55fb75d3` (Gemma/Nemotron probes).
- Results on disk: `results/<model>/resample/*/scores.json` (DeepSeek 128 on DeepInfra at S=20 and S=50, Qwen 768 on Chutes stride 8, Inkling 7979 on Together S=30) and the `provider` field on every replay record.
