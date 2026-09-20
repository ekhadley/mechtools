# mechtools

Shared prelude for the mechinterp research projects in `~/wgmn`. See README.md for the module layout and what was deliberately left out.

## Not in the package, reference copies elsewhere

- SAE helpers (`load_sae`, `save_sae`, `top_feats_summary`, `get_sae_pre_acts`, `get_latent_dec`, neuronpedia dashboard links): `subliminal_learning/utils.py`, `anonymous_subliminal/utils.py`, `ao/utils.py` (load/save/top_feats), `sae_lora/utils.py` (`get_sae_pre_acts`, `top_feats_summary`), `introspect/utils.py` (`get_latent_dec`).
- LLM judges: not in mechtools. Projects keep their own on top of `mechtools.openrouter.chat` (reference shapes: `RubricJudge` in `weirdchat/weirdchat/judge.py`, `value-leakage/src/value_leakage/judge.py`).

## API runs

- `OPENROUTER_API_KEY` comes from the project's `.env` through `load_dotenv()` at import. This repo has no `.env`: a live check from here loads a project's (`load_dotenv("/home/ek/wgmn/weirdchat/.env")` in a scratch script). Never print the key.
- Live calls cost money. The tests use fakes; keep any live check to a few requests with small `max_tokens`, and report the spend.
- Batch helpers go through `gather_bar`, whose status line separates slow from throttled, counts failed attempts by cause, and shows dollars. A coroutine that exhausts its retries yields `None` and a rerun fills the deficit; auth and credit errors stop the batch.
- CoT resampling: run `mechtools.resample.probe` on the model's endpoints before any paid run, and read the `mechtools.resample` module docstring, which is the checklist of what to verify, what each failure looks like, and the pitfalls. The measurements behind it (weirdchat, 2026-09) are in `docs/openrouter_provider_findings.md`; rosters change, so probe rather than trust the file.
