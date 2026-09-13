# mechtools

Shared helpers for mechinterp research projects, importable locally (not published).

## Install

Per project, from that project's directory:

```
uv add --editable ~/wgmn/mechtools
```

Then `from mechtools import *` at the top of a script gives the whole prelude: color constants, `tec`, `set_seed`, and everything below. Importing also turns on IPython autoreload when in a kernel and calls `load_dotenv()`.

## Layout

| Module | Contents |
|---|---|
| `colors` | Terminal color escape constants |
| `tokens` | `to_str_toks`, `underline_stoks`, `find_first_idx`, `get_turn_tok_idx`, `add_system_prompt_to_messages`, `apply_chat_template` (single or left-padded batch), `get_assistant_mask`, `completion_loss` |
| `tables` | `top_toks_table`, `show_table`, `html_table`, `print_titled_table`. Tables render as HTML in a notebook kernel and as tabulate text elsewhere |
| `hooks` | `add_bias_hook` (`scale`, `seq_pos`, `target_norm`), `make_add_bias_hook`, `replace_act_hook`, `make_sae_feat_steer_hook`, `proj_out`, `scale_hooks`, `set_hooks` |
| `sampling` | `stream_toks`, `stream_toks_hf`, `sample_batch`, `sample_rolling` for a `TransformerBridge` |
| `models` | `load_hf_model` (peft adapter auto-detect and merge), `load_bridge` (`TransformerBridge.boot_transformers` around it, eval, grads off) |
| `stats` | `normed`, `cosine_sim`, `pearson`, `mean_self_sim`, `topk_vector_matches`, `kmeans`, `wilson` |

Models are `TransformerBridge` objects (transformer-lens >= 3.8); there is no `HookedTransformer` path. `seq_pos` arguments take an int, a slice, or a list of ints.

## Standardization candidates

From a scan of the 24 user-authored Python research projects in `~/wgmn`. Sorted by breadth of duplication times how identical the copies already are. "Projects" is how many define or use the thing.

| # | Candidate | Projects | Uniformity |
|---|---|---|---|
| 1 | Terminal color constants, `tec()`, IPython autoreload block | 15 / 9 / 17 | Byte-identical apart from a few extra colors (`brown`, `magenta`, `white`) in newer copies |
| 2 | `plotly_utils.py` (`imshow`, `line`, `scatter`, `bar`, `hist`, `to_numpy`) | 6 file copies + 4 inlined older versions | 3 copies are md5-identical, GLP differs by 5 lines, ioi_replication and mech are stale forks |
| 3 | `top_toks_table` / `topk_toks_table`, `print_titled_table`, `top_templates_table` | 13 | Same idea, 4 signature variants (`k` default 10 vs 25, `show_probs`, `return_top`, `initial_logits`, topk-object input) |
| 4 | OpenRouter batch querying + results json save/load | 7 | Two lineages: sync `requests`+`ThreadPoolExecutor` (value-leakage, odd-number-hacking, near-identical) and async `AsyncOpenAI`+tenacity+semaphore (safety-refusals and value-leakage/src/api are copies). Plus a sqlite response cache in safety-refusals |
| 5 | Tokenizer string helpers: `to_str_toks` (3 different names), `underline_stoks`, `get_turn_tok_idx`, `find_first_idx`, `apply_chat_template` wrapper, `add_system_prompt_to_messages`, `get_assistant_mask`, `completion_loss` | 7 to 19 | `get_assistant_mask` copied verbatim 3x. Raw `apply_chat_template` in 19 projects |
| 6 | J-lens / template-lens loading and readout (`load_jlens`, `load_tlens`, `jlens_transport`, `get_lens_logits`, `get_template_*`, `get_tlens_scores`) plus the Flask `lens.py` viewer | 4 (+2 for viewer) | Near-identical. Arguably belongs in the `jacobian-lens` package rather than here |
| 7 | Local sampling: `stream_toks` (Bridge and HF variants), `sample_batch`, `sample_rolling`, and the `TransformerBridge.boot_transformers` + seed + `set_grad_enabled(False)` header in `local_tl.py` | 3 to 4 | `stream_toks` identical in 3. `sample_rolling` is weirdchat-only but fully general |
| 8 | Steering and patching hooks: `add_bias_hook`, `make_sae_feat_steer_hook`, `replace_act_hook`, `scale_hook`/`set_hook` projection hooks, `proj_out` | 14 define some `*_hook` | Two `add_bias` variants (seq_pos-aware vs target_norm). Easy to merge into one signature |
| 9 | Model loading: `load_hf_model_with_adapter` (peft auto-detect + merge), `load_hf_model_into_hooked` | 2 verbatim, 12 use `HookedTransformer.from_pretrained`, 12 use `AutoModelForCausalLM`, 4 use Bridge | One `load_model(id, kind=...)` that handles adapter detection, dtype, device, eval, and `requires_grad_(False)` covers most call sites |
| 10 | `set_seed` | 12 | Defined once, elsewhere a 3-line np/random/torch block |
| 11 | SAE helpers: `load_sae`/`save_sae`, `top_feats_summary`, neuronpedia dashboard link, `get_latent_dec`, `get_sae_pre_acts` | 6 | `load_sae` copied 2x, the others 3x with small diffs |
| 12 | Small tensor stats: `pearson`, `cosine_sim`, `topk_vector_matches`, `get_mean_self_sim`, `kmeans`, `normed`, `wilson` CI | 4 | Tiny, no design decisions |
| 13 | Progress bar and wandb conventions: `tqdm(..., ncols=120, ascii=" >=", desc=colored)`, `wandb.init(project, name, config=asdict)`, `load_dotenv()` at import | 9 / 7 / 11 | A `pbar()` wrapper and a `wandb_init()` are marginal wins. `load_dotenv` on `mechtools` import is a free one |
| 14 | LLM judges (`RubricJudge`, value-leakage `_judge`, subliminal `Judge`, sae_lora classifier) | 4 | Same shape: prompt template, json/tag parse, async batch. All different prompts. Only worth doing after #4 |
| 15 | Activation harvesting and storage (subliminal act_store, avae `act_hoard`, GLP `MemmapWriter`, neural_chameleons `ActivationCache`, agent-interp-envs `capture`, jlens `ActivationRecorder`) | 6 | Six different designs. This is a design project, not an extraction. Skip for now |
| 16 | Flask token viewers (`view_tokens.py` x2, `view.py`, `view_run.py`) | 4 | Each is bespoke to its data format. Only the token-strip HTML/JS is shareable |

Rows 1, 3, 5, 7, 8, 9, 10, and 12 are in the package. Rows 2, 4, 6, 11, 13, and 14 are next. Rows 15 and 16 are left out for now.
