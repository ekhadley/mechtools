# mechtools

Shared helpers for mechinterp research projects, importable locally (not published).

## Install

Per project, from that project's directory:

```
uv add --editable ~/wgmn/mechtools
```

Then `from mechtools import *` at the top of a script gives the whole prelude: color constants, `tec`, `set_seed`, `pbar` (tqdm with colored desc, `ncols=120`, ascii fill), and everything below. Importing also turns on IPython autoreload when in a kernel and loads the first `.env` found walking up from the current directory, the project's when run from the project directory; that is where `OPENROUTER_API_KEY` goes, and a missing key raises a `RuntimeError` that names the file it looked at.

## Tests

```
./test.sh
```

The tokenizer tests load Qwen3 (0.6B, 8B, 3.6-27B, 3.8-27B), Qwen2.5-3B-Instruct, gemma-3 (1b, 4b), Llama-3 (3.2-1B, 3.1-8B) and Starling-LM-7B-alpha tokenizers from the local HF cache. The `openrouter` and `resample` tests run on fakes and make no API calls; the live checks are `probe` and `sampling_defaults` in `mechtools.resample`, which spend a few cents.

## Layout

| Module | Contents |
|---|---|
| `colors` | Terminal color escape constants |
| `tokens` | `to_ids`, `to_str_toks`, `show_toks` (hoverable HTML token strip); all take a string, ids, or a conversation. `get_turn_tok_idx` (token span of one message), `apply_chat_template` (left-padded batch of conversations or prompt strings), `get_assistant_mask` (ids, attention mask, and a mask over assistant tokens), `completion_loss`. Tested against Qwen3, Qwen2.5, gemma-3, Llama-3 and Starling (sentencepiece) templates; gemma-2 templates reject system messages and are not supported |
| `tables` | `top_toks_table`, `show_table`, `html_table`, `print_titled_table`. Tables render as HTML in a notebook kernel and as tabulate text elsewhere |
| `lens` | `load_jlens`, `load_tlens`, `jlens_transport`, `get_lens_logits`, `get_jlens_token_vec`, `get_template_vec(s)`, `get_tlens_scores`, `print_templates`, `top_templates_table`; HTML readouts `readout_html` (dark frame with a token strip), `readout_grid`, `tabbed` (one or two tab bars over panes stacked in one grid cell, so the widget keeps the height of its tallest pane and does not resize as tabs change; after clicking the widget, left/right and up/down arrow keys switch tabs), `top_readout`, `jlens_readout`, `tlens_readout`; cluster readouts `cluster_readout`, `jlens_cluster_readout`, `tlens_cluster_readout` (a tab per layer, and a second bar of tabs when `pos` is a list; one token strip above the bars marks every position read out and clicking a marked token switches to it), `cluster_tlens`, `vocab_vecs` and `cluster_vocab` (k-means over the mean-centered unembedding, or embedding with `embed=True`). Readouts take `input_src` for the token strip: str tokens, a string, ids, or a conversation |
| `hooks` | `add_bias_hook` (`scale`, `seq_pos`, `target_norm`), `make_add_bias_hook`, `replace_act_hook`, `make_sae_feat_steer_hook`, `proj_out`, `scale_hooks`, `set_hooks` |
| `sampling` | `stream_toks`, `stream_toks_hf`, `sample_batch`, `sample_rolling` for a `TransformerBridge` |
| `openrouter` | Async OpenRouter requests: `chat` (chat completions; `reasoning` on/off or an effort, `provider` pin) and `complete` (raw `/completions` on a self-rendered prompt string, for CoT resampling). Both retry with backoff on 408/429/5xx, timeouts, error envelopes and a `finish_reason` of `error`; any other 4xx raises `RequestFailed` at once, and 401/402 (no key, no credits) raise `RuntimeError`, which stops a batch. They return the record from `flat` (`text`, `reasoning`, `finish_reason`, `provider`, token counts, `cost`) or the body with `raw=True`. `gather_bar` awaits coroutines at bounded concurrency; a failed one yields `None`, and the status line shows ok/fail, dollars, open requests with the age of the oldest, seconds per request, backoff, truncations, and failed attempts by cause, then a summary line. `chat_batch`, `complete_batch`, `endpoints` (the providers serving a model, with the slug to pin) |
| `resample` | `Resampler`: token-level resampling of a CoT or response through raw `/completions` on a passthrough provider. `prefix(t)` (None where the cut does not retokenize), `grid(stride)`, `rollout(t, i)` (one call, or two with `stop` and `response_open`; raises when the provider's `prompt_tokens` disagree with the cut), `fill({t: count})` (tops up the per-position deficit in a rollouts jsonl under `gather_bar`, so reruns and adaptive schemes are more `fill` calls), `scores(key)` (per-position p with Wilson interval from the judge's key), `resample_curve` (plotly p-over-t with the band). `probe` checks every endpoint of a model for raw-prompt passthrough before a paid run: token-count offsets, whether continuations continue or restart, split or merged return path, leaked special tokens, the provider's completion-token constant. `sampling_defaults` measures whether a provider truncates sampling when top_p and top_k are omitted. The module docstring is the checklist: what to verify, what each failure looks like, and the pitfalls. The measurements behind it, from weirdchat: `docs/openrouter_provider_findings.md` |
| `models` | `load_hf_model` (peft adapter auto-detect and merge), `load_bridge` (`TransformerBridge.boot_transformers` around it, eval, grads off) |
| `stats` | `normed`, `cosine_sim`, `pearson`, `mean_self_sim`, `topk_vector_matches`, `kmeans`, `hierarchical_kmeans` (k-means then cosine agglomeration of the centroids), `wilson` |
| `plots` | `imshow`, `line`, `scatter`, `bar`, `hist`, `to_numpy` plotly wrappers; `plot_vocab_umap` (UMAP scatter of a token-vector subset colored by cluster). The general half of the shared `plotly_utils.py`; the ARENA task-specific plots are not included |

Models are `TransformerBridge` objects (transformer-lens >= 3.8); there is no `HookedTransformer` path. `seq_pos` arguments take an int, a slice, or a list of ints.

## Standardization candidates

From a scan of the 24 user-authored Python research projects in `~/wgmn`. Sorted by breadth of duplication times how identical the copies already are. "Projects" is how many define or use the thing.

| # | Candidate | Projects | Uniformity |
|---|---|---|---|
| 1 | Terminal color constants, `tec()`, IPython autoreload block | 15 / 9 / 17 | Byte-identical apart from a few extra colors (`brown`, `magenta`, `white`) in newer copies |
| 2 | `plotly_utils.py` (`imshow`, `line`, `scatter`, `bar`, `hist`, `to_numpy`) | 6 file copies + 4 inlined older versions | In `mechtools.plots`, from the copy shared by jlens_fun, agent-interp-envs and odd-number-hacking |
| 3 | `top_toks_table` / `topk_toks_table`, `print_titled_table`, `top_templates_table` | 13 | Same idea, 4 signature variants (`k` default 10 vs 25, `show_probs`, `return_top`, `initial_logits`, topk-object input) |
| 4 | OpenRouter batch querying + results json save/load | 7 | In `mechtools.openrouter`, rewritten over raw httpx rather than either lineage: weirdchat's async chat and raw-completions calls, retry policy, and progress bar. Results save/load and the sqlite cache stay in projects; the resampler's per-position deficit loop is in `mechtools.resample` |
| 5 | Tokenizer string helpers: `to_str_toks` (3 different names), `underline_stoks`, `get_turn_tok_idx`, `find_first_idx`, `apply_chat_template` wrapper, `add_system_prompt_to_messages`, `get_assistant_mask`, `completion_loss` | 7 to 19 | In `mechtools.tokens`, rewritten around a sentinel-token span finder rather than copied; the copied `get_assistant_mask` was wrong on Qwen3 and off by one elsewhere |
| 6 | J-lens / template-lens loading and readout (`load_jlens`, `load_tlens`, `jlens_transport`, `get_lens_logits`, `get_template_*`, `get_tlens_scores`) plus the Flask `lens.py` viewer | 4 (+2 for viewer) | In `mechtools.lens`, with the Flask viewer replaced by HTML readouts. Clustered readouts share the frame |
| 7 | Local sampling: `stream_toks` (Bridge and HF variants), `sample_batch`, `sample_rolling`, and the `TransformerBridge.boot_transformers` + seed + `set_grad_enabled(False)` header in `local_tl.py` | 3 to 4 | `stream_toks` identical in 3. `sample_rolling` is weirdchat-only but fully general |
| 8 | Steering and patching hooks: `add_bias_hook`, `make_sae_feat_steer_hook`, `replace_act_hook`, `scale_hook`/`set_hook` projection hooks, `proj_out` | 14 define some `*_hook` | Two `add_bias` variants (seq_pos-aware vs target_norm). Easy to merge into one signature |
| 9 | Model loading: `load_hf_model_with_adapter` (peft auto-detect + merge), `load_hf_model_into_hooked` | 2 verbatim, 12 use `HookedTransformer.from_pretrained`, 12 use `AutoModelForCausalLM`, 4 use Bridge | One `load_model(id, kind=...)` that handles adapter detection, dtype, device, eval, and `requires_grad_(False)` covers most call sites |
| 10 | `set_seed` | 12 | Defined once, elsewhere a 3-line np/random/torch block |
| 11 | SAE helpers: `load_sae`/`save_sae`, `top_feats_summary`, neuronpedia dashboard link, `get_latent_dec`, `get_sae_pre_acts` | 6 | Left out. CLAUDE.md lists the projects to copy from |
| 12 | Small tensor stats: `pearson`, `cosine_sim`, `topk_vector_matches`, `get_mean_self_sim`, `kmeans`, `normed`, `wilson` CI | 4 | Tiny, no design decisions |
| 13 | Progress bar and wandb conventions: `tqdm(..., ncols=120, ascii=" >=", desc=colored)`, `wandb.init(project, name, config=asdict)`, `load_dotenv()` at import | 9 / 7 / 11 | `pbar` and `load_dotenv` on import are in. No wandb wrapper |
| 14 | LLM judges (`RubricJudge`, value-leakage `_judge`, subliminal `Judge`, sae_lora classifier) | 4 | Left out. Same shape but all different prompts, and depends on #4 |
| 15 | Activation harvesting and storage (subliminal act_store, avae `act_hoard`, GLP `MemmapWriter`, neural_chameleons `ActivationCache`, agent-interp-envs `capture`, jlens `ActivationRecorder`) | 6 | Six different designs. This is a design project, not an extraction. Skip for now |
| 16 | Flask token viewers (`view_tokens.py` x2, `view.py`, `view_run.py`) | 4 | Each is bespoke to its data format. Only the token-strip HTML/JS is shareable |

Rows 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, and 13 are in the package. Rows 11, 14, 15, and 16 are left out.
