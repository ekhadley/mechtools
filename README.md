# mechtools

Shared helpers for mechinterp research: tokenization and chat-template utilities, local sampling, steering hooks, lens readouts rendered as HTML in the notebook, async OpenRouter batching, chain-of-thought resampling, and plotly wrappers. Models are `TransformerBridge` objects from transformer-lens 3.

## Install

From a project's directory:

```
uv add git+https://github.com/ekhadley/mechtools
```

or, from a local clone, `uv add --editable path/to/mechtools`.

Then:

```python
from mechtools import *
```

This imports every module's public names plus the prelude: terminal color constants, `pbar` (tqdm in the house style), `set_seed`, and `tec` (empties the CUDA cache). Inside an IPython kernel it also turns on autoreload. It loads the first `.env` found walking up from the current directory, which is where `OPENROUTER_API_KEY` goes; import succeeds without one, and the first OpenRouter request tells you which `.env` was loaded if the key is missing.

## Quick tour

```python
from mechtools import *

model = load_bridge("Qwen/Qwen3-0.6B")      # TransformerBridge, eval mode, grads off; peft adapter repos are merged automatically
tok = model.tokenizer
conv = [{"role": "user", "content": "What is 17 * 23?"}]

show_toks(conv, tok, add_generation_prompt=True, enable_thinking=False)      # hoverable token strip
ids, attn = apply_chat_template(tok, conv, enable_thinking=False)            # left-padded [batch, seq]
print(tok.decode(list(stream_toks(model, ids, new_toks=64))))                # sample until eos

show_logits(conv, model=model, k=10)        # click any token to see the top-k predictions after it
```

## Modules

### `tokens`

Every function takes the same kinds of input: a string, a sequence of token ids, or a conversation (a list of role/content dicts, rendered through the tokenizer's chat template). Chat-template kwargs like `add_generation_prompt` or `enable_thinking` are forwarded.

- `to_ids`, `to_str_toks`: ids or decoded strings of any input.
- `show_toks`: HTML token strip. Hover shows index, id and repr. `pos` underlines one token. `vals=` shades each token by a scalar (red positive, blue negative), for attributions or probe scores along a sequence.
- `underline_stoks`: the same boundaries in the terminal, alternating underline.
- `apply_chat_template`: a left-padded `(input_ids, attention_mask)` batch from a list of conversations or plain prompt strings.
- `get_turn_tok_idx`: token span of one message's content inside the rendered conversation.
- `get_assistant_mask`: `(input_ids, attention_mask, assistant_mask)` where the mask is 1 on the tokens an SFT loss should predict, the content of every assistant turn plus its end-of-turn token.
- `completion_loss`: mean cross-entropy over the masked tokens.

```python
convs = [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], ...]
ids, attn, mask = get_assistant_mask(tok, convs)
loss = completion_loss(model(ids), ids, mask)
```

Tested against the Qwen3, Qwen2.5, gemma-3, Llama-3 and Starling templates. When you pass a string you rendered yourself from a chat template, pass `add_special_tokens=False` so BOS is not added twice.

### `sampling`

Temperature-1 sampling from a `TransformerBridge` (or a raw HF model with `stream_toks_hf`). Everything stops at any of the model's end-of-sequence ids, which are read from both the generation config and the tokenizer, since chat models often end a turn with a token that is not the tokenizer's eos.

- `stream_toks`: yields one token id at a time.
- `sample_batch`: `n` independent samples of one prompt as a single batch.
- `stream_rolling` / `sample_rolling`: `n` samples with a fixed batch size, yielding each as it finishes and refilling its slot. Use this for large `n`; a run that saves each sample as it arrives survives a crash, and calling again with the remaining `n` tops it up.

```python
for sample in stream_rolling(model, ids, n=2000, batch_size=64, new_toks=512):
    out.write(json.dumps(tok.decode(sample)) + "\n")
```

A returned sample of length `new_toks` hit the cap without stopping.

- `stream_rollouts`: continuations of one text from many cut positions, for resampling a local model. `cuts[i]` is how many tokens of `toks` rollout i keeps, prompt included. One forward over the text, then every rollout starts from a right-aligned slice of that cache, so no prefix is recomputed; rows run in batches from the longest cut down and leave their batch as they finish, and samples arrive in completion order as `(i, sample)`. `max_len` caps every row at the same total length wherever it was cut, so a rollout's distribution does not depend on its cut; a sample of `max_len - cuts[i]` tokens hit it. Full-attention models only. Since the long rollouts of a position are the ones still running when a run dies, save a position's rollouts together once all of them are in, not one by one, or a resumed pool is biased toward short continuations there.

```python
cuts = [n_prompt + t for t in range(T + 1) for _ in range(10)]   # 10 rollouts from every position of a T-token response after an n_prompt-token prompt
for i, sample in stream_rollouts(model, ids, cuts, batch_size=64, max_len=n_prompt + 1024):
    out.write(json.dumps({"t": cuts[i] - n_prompt, "ids": sample}) + "\n")
```

`resample.estimate` turns such records into the curve; see that section.

### `hooks`

Hook functions for `model.hooks(fwd_hooks=...)` and `model.run_with_hooks`.

- `add_bias_hook` / `make_add_bias_hook`: add a vector, optionally scaled, rescaled to a `target_norm`, or only at `seq_pos`.
- `replace_act_hook`: overwrite the activation, or the positions in `seq_pos`.
- `make_sae_feat_steer_hook`: `(hook_name, hook)` that adds a multiple of an SAE feature's decoder direction.
- `scale_hooks` / `set_hooks`: per-layer hooks that rescale (0 ablates) or set the residual's projection onto a direction or a set of directions. The set is orthonormalized first, so repeated or correlated directions are not double counted.
- `proj_out`: remove the component of a vector, or each row of a stack, along a direction.

```python
hooks = scale_hooks({8: refusal_dir, 12: refusal_dir}, factor=0.0)
with model.hooks(fwd_hooks=hooks):
    logits = model(ids)

steer = ("blocks.8.hook_resid_pre", make_add_bias_hook(vec, scale=4.0, seq_pos=slice(-3, None)))
logits = model.run_with_hooks(ids, fwd_hooks=[steer])
```

`seq_pos` is an int, a slice, or a list of ints everywhere.

### `lens`

Loading and readouts for the j-lens and template-lens from the `camilablank/workspace-lenses` Hub repo, plus HTML readout widgets that work for any per-token or per-template scores.

```python
_, cache = model.run_with_cache(ids)
jlens = load_jlens("qwen3.6-27b/j-lens/lens.pt", device=model.device)
tlens = load_tlens("qwen3.6-27b/template-lens/templates+phrases_v3.safetensors")

jlens_readout(cache, layers=[8, 16, 24], pos=-1, model=model, jlens=jlens, input_src=ids)    # top tokens per layer
tlens_readout(cache, layers=[8, 16, 24], pos=-1, tlens=tlens, input_src=ids, tokenizer=tok)  # top templates per layer

labels, _ = cluster_vocab(model, k=1024)
jlens_cluster_readout(cache, layers=[8, 16, 24], pos=[-1, -5], model=model, jlens=jlens, labels=labels, input_src=ids)
```

- `show_logits`: top-k next-token table for every position of an input, one at a time. Click a token in the strip, or use the arrow keys, to switch position. The row of the actual next token is highlighted. Pass `model` to run it, or `logits` from your own forward pass.
- `jlens_readout`, `tlens_readout`: one top-k table per layer at a position.
- `jlens_cluster_readout`, `tlens_cluster_readout`: the same with the vocabulary or the templates clustered by k-means, so a diffuse readout shows as a few named clusters instead of a long tail. A tab per layer and, when `pos` is a list, a second tab bar per position.
- `top_readout`, `cluster_readout`: the underlying widgets, for scores you computed yourself (`{header: [n] scores}` plus a names list or a decode function).
- `get_lens_logits`, `get_tlens_scores`, `get_jlens_token_vec`, `get_template_vec`, `cluster_vocab`, `cluster_tlens`: the computations behind them.

Readouts take `input_src` for the token strip: a string, ids, or a conversation. A string is tokenized with special tokens added, so for a self-rendered template string pass its ids.

### `tables`

`top_toks_table(logits, tok, k=10)` shows the k most likely tokens of one position's logits (`show_negative=True` adds the least likely). `show_table(headers, rows)` renders any rows. Both are HTML in a notebook kernel and a tabulate text table elsewhere.

### `openrouter`

Async requests with retries, a progress bar, and cost tracking. Requires `OPENROUTER_API_KEY` in a `.env` or the environment.

```python
body = await chat("What is 17 * 23?", "qwen/qwen3-30b-a3b", reasoning=False, max_tokens=64)
print(flat(body)["text"], flat(body)["cost"])

bodies = await chat_batch(convs, "qwen/qwen3-30b-a3b", concurrency=16, desc="judge")   # list, None where a request failed
```

- `chat`: one chat completion. `reasoning` is on/off or an effort string; `provider` pins one endpoint by slug. Extra kwargs go into the request body.
- `complete`: one raw `/completions` call on a prompt string you rendered yourself.
- `flat`: the common fields of a response body (`text`, `reasoning`, `finish_reason`, `provider`, token counts, `cost`). Store the whole body; it has more.
- `chat_batch`, `complete_batch`: many requests at bounded concurrency, results in order.
- `gather_bar`: the batching loop itself, for your own coroutines. Its status line shows requests done and failed, dollars spent and projected, open requests and the age of the oldest, backoff, and failed attempts by cause.
- `endpoints`: the providers serving a model, with the slug to pin.

Retries cover network errors, 408/429/5xx, timeouts and error envelopes. Other 4xx errors raise `RequestFailed`, which a batch records as `None` and moves on. A missing key or empty credits raises `RuntimeError` and stops the batch, and so does a run where the first 8 requests all fail, which usually means a bad model or parameter. Batch-stopping errors arrive wrapped in an `ExceptionGroup` (the batch runs under `asyncio.TaskGroup`), so catch them with `except*`.

### `resample`

Token-level resampling of a chain of thought or a response through raw `/completions`: cut the text at token positions, sample continuations from each prefix, judge them, and get P(outcome | prefix) per position with Wilson intervals.

```python
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)   # ends inside the open think block
rs = Resampler(tok, prompt, cot, "rollouts.jsonl", model="qwen/qwen3-30b-a3b", provider="chutes", judge=judge, max_tokens=8192, top_p=1.0, top_k=0)
await rs.fill({t: 10 for t in rs.grid(stride=1)})     # tops up each position to 10 rollouts on disk, then judges them; rerun to fill failures
while d := rs.deficit(0.1, n_min=20):                  # early stopping: 5 more wherever the interval is still wider than +-0.1 or fewer than 20 samples are in
    await rs.fill(d)
resample_curve(rs.scores("match"))
```

`judge` is an async function from a rollout record to a dict of fields to store with it; `scores(key)` reads one of those keys. Rollouts append to a jsonl as they finish and verdicts to a sidecar next to it, so a judge outage costs nothing: the next `fill` judges what is pending. Loading the file checks every record against the instance's configuration, so runs cannot be mixed.

`scores(key, method)` has three estimators. `reuse` (the default) counts a rollout at every position it carries the text through: a continuation from t that reproduces the text's next k tokens is a sample from t + k as well, exactly. On a 165-token math response that gave 7.9 effective samples per rollout, which is why a stride of 1 at a small count per position beats a coarse grid at a large one, and why `deficit` measures its floor and interval on the effective counts. `recursion` reads the same counts backward from the end, p(t) = q p(t+1) + (1 - q) r with q the share of rollouts at t that continue with the text's next token and r the hit rate of those that leave, so rollouts sampled after t inform p(t) too; on math traces it needs about half the tokens of reuse at equal error, with a calibrated interval. `naive` uses only the rollouts sampled at t.

The same estimators are `estimate(T, rollouts, method)` over `(t, k, outcome)` triples from anywhere, with k the number of the text's next tokens the rollout reproduced: rollouts of a local model from `sampling.stream_rollouts`, for instance. Reuse and the recursion are exact only when a rollout's distribution does not depend on where it was cut, so the length cap must rarely bind: `Resampler` counts its max_tokens from the cut, and `sampling.stream_rollouts` caps the total length for that reason.

This only works when the provider feeds the model exactly the string you send, and nothing in a response says whether it did. Run `probe(tok, model_id, render)` on the model's endpoints first; it costs a few cents and reports which providers pass the prompt through verbatim. The `mechtools.resample` module docstring is the full checklist of what to verify and what each failure looks like. Read it before spending money.

### `models`

- `load_bridge(model_id)`: a `TransformerBridge` in eval mode with grads off. If `model_id` is a peft adapter repo, the base model is loaded and the adapter merged in memory.
- `load_hf_model(model_id)`: the same as a raw HF model.
- `is_adapter_repo(model_id)`: whether a local directory or Hub repo holds an `adapter_config.json`.

### `stats`

`cosine_sim`, `pearson`, `normed`, `mean_self_sim`, `topk_vector_matches`, `kmeans` (spherical, stops when labels settle), `hierarchical_kmeans`, and `wilson(k, n)` confidence intervals.

### `plots`

`imshow`, `line`, `scatter`, `bar`, `hist`: plotly wrappers that take tensors, arrays or lists directly, plus `renderer` and `return_fig`. `to_numpy` converts anything tensor-like. `plot_vocab_umap` scatters a subset of token vectors colored by cluster.

## Tests

```
./test.sh
```

Runs pytest offline. Tests that need a tokenizer or model from the local HF cache skip when it is absent; `./test.sh -m "not hf"` runs the rest anywhere. The integration tests use `hf-internal-testing/tiny-random-LlamaForCausalLM` (1M parameters), which `hf download hf-internal-testing/tiny-random-LlamaForCausalLM` puts in the cache. No test makes an API call.

## Not included

SAE loading and feature helpers, LLM judges, and activation harvesting stay in the individual projects. Judges are a few lines on top of `openrouter.chat` with a project-specific prompt.
