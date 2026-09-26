# mechtools

Shared prelude for the mechinterp research projects in `~/wgmn`. See README.md for the module layout and what was deliberately left out.

## Not in the package, reference copies elsewhere

- SAE helpers (`load_sae`, `save_sae`, `top_feats_summary`, `get_sae_pre_acts`, `get_latent_dec`, neuronpedia dashboard links): `subliminal_learning/utils.py`, `anonymous_subliminal/utils.py`, `ao/utils.py` (load/save/top_feats), `sae_lora/utils.py` (`get_sae_pre_acts`, `top_feats_summary`), `introspect/utils.py` (`get_latent_dec`).
- LLM judges: not in mechtools. Projects keep their own on top of `mechtools.openrouter.chat` (reference shapes: `RubricJudge` in `weirdchat/weirdchat/judge.py`, `value-leakage/src/value_leakage/judge.py`).

## API runs

- `OPENROUTER_API_KEY` comes from the first `.env` found walking up from the cwd when mechtools is imported: the project's when run from its directory. This repo has no `.env`, so from here the walk continues to `~/wgmn` and `~`. A missing key raises a `RuntimeError` naming the file it looked at. A variable already set in the environment keeps its value (a shell export or a command-line `VAR=...` wins over the file) and import prints a note naming it. Never print the key.
- Live calls cost money. The tests use fakes; keep any live check to a few requests with small `max_tokens`, and report the spend.
- Batch helpers go through `gather_bar`, whose status line separates slow from throttled, counts failed attempts by cause, and shows dollars. A coroutine that exhausts its retries yields `None` and a rerun fills the deficit; auth and credit errors stop the batch, and so does a batch whose first 8 coroutines fail before any succeeds (`abort_after`): a bad model, provider or parameter, or an endpoint that is down.
- CoT resampling: run `mechtools.resample.probe` on the model's endpoints before any paid run, and read the `mechtools.resample` module docstring, which is the checklist of what to verify, what each failure looks like, and the pitfalls. The measurements behind it (weirdchat, 2026-09) are in `docs/openrouter_provider_findings.md`; rosters change, so probe rather than trust the file.

## Todo

- The advanced readout functions (`show_logits`, `jlens_readout`, `tlens_readout`, and the other `top_readout` callers) should return a packaged object holding both the rendered visualization and the underlying computed data, instead of only displaying. That object would carry `save_html` for sharing a readout outside the notebook, methods to turn the same data into plots of various forms, and plain attribute access to the scores/tokens that were computed.
