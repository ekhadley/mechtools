# mechtools

Shared prelude for the mechinterp research projects in `~/wgmn`. See README.md for the module layout and what was deliberately left out.

## Not in the package, reference copies elsewhere

- SAE helpers (`load_sae`, `save_sae`, `top_feats_summary`, `get_sae_pre_acts`, `get_latent_dec`, neuronpedia dashboard links): `subliminal_learning/utils.py`, `ao/utils.py` (load/save/top_feats), `sae_lora/utils.py` (`get_sae_pre_acts`, `top_feats_summary`), `introspect/utils.py` (`get_latent_dec`).
- LLM judges: not in mechtools. Projects keep their own on top of `mechtools.openrouter.chat` (reference shapes: `RubricJudge` in `weirdchat/weirdchat/judge.py`, `value-leakage/src/value_leakage/judge.py`).

## Tests

- `./test.sh` runs pytest offline against the local HF cache. Tests that need a tokenizer or model from the cache carry the `hf` marker (added automatically to any test using the `tok`, `qwen` or `tiny_bridge` fixtures) and skip when it is absent; `./test.sh -m "not hf"` runs the rest anywhere. The integration tests of sampling, hooks, readouts and loading use `hf-internal-testing/tiny-random-LlamaForCausalLM` through the `tiny_bridge` fixture in `tests/conftest.py`; `hf download hf-internal-testing/tiny-random-LlamaForCausalLM` puts it in the cache.
- Network code is tested against `httpx.MockTransport` and fakes of `openrouter._post` and `openrouter.complete`; no test makes an API call.

## API runs

- `OPENROUTER_API_KEY` comes from the first `.env` found walking up from the cwd when mechtools is imported: the project's when run from its directory. This repo has no `.env`, so from here the walk continues to `~/wgmn` and `~`. Import succeeds without a key; the first request then raises a `RuntimeError` naming the `.env` that was loaded (or the cwd when none was found). A variable already set in the environment keeps its value (a shell export or a command-line `VAR=...` wins over the file), and when that value differs from the file's, import prints a note naming it. Never print the key.
- Live calls cost money. The tests use fakes; keep any live check to a few requests with small `max_tokens`, and report the spend.
- Batch helpers go through `gather_bar`, whose status line separates slow from throttled, counts failed attempts by cause, and shows dollars. A coroutine that exhausts its retries yields `None` and a rerun fills the deficit; auth and credit errors stop the batch, and so do 8 failed coroutines before the first success (`abort_after`): a bad model, provider or parameter, or an endpoint that is down. A batch-stopping error, like any exception other than `RequestFailed`, reaches the caller wrapped in an `ExceptionGroup` (`gather_bar` runs under `asyncio.TaskGroup`); catch it with `except*`.
- CoT resampling: run `mechtools.resample.probe` on the model's endpoints before any paid run, and read the `mechtools.resample` module docstring, which is the checklist of what to verify, what each failure looks like, and the pitfalls. The measurements behind it (weirdchat, 2026-09) are in `docs/openrouter_provider_findings.md`; rosters change, so probe rather than trust the file. That file describes weirdchat's resampler where it names file formats, modes or assertions; `mechtools.resample.Resampler` is documented in its own docstrings.

## Todo

- The advanced readout functions (`show_logits`, `top_readout` and its callers `jlens_readout` and `tlens_readout`, and the cluster readouts) should return a packaged object holding both the rendered visualization and the underlying computed data, instead of only displaying. That object would carry `save_html` for sharing a readout outside the notebook, methods to turn the same data into plots of various forms, and plain attribute access to the scores/tokens that were computed.
