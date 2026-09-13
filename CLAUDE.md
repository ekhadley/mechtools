# mechtools

Shared prelude for the mechinterp research projects in `~/wgmn`. See README.md for the module layout and what was deliberately left out.

## Not in the package, reference copies elsewhere

- SAE helpers (`load_sae`, `save_sae`, `top_feats_summary`, `get_sae_pre_acts`, `get_latent_dec`, neuronpedia dashboard links): `subliminal_learning/utils.py`, `anonymous_subliminal/utils.py`, `ao/utils.py` (load/save/top_feats), `sae_lora/utils.py` (`get_sae_pre_acts`, `top_feats_summary`), `introspect/utils.py` (`get_latent_dec`).
- API sampling and LLM judges: not in mechtools. Projects keep their own.
