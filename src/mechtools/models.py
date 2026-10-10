import os

import torch as t
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from peft import PeftConfig, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.base import GeneralizedComponent
from transformers import AutoModelForCausalLM

from mechtools.colors import gray, orange, endc

def is_adapter_repo(model_id: str) -> bool:
    """Whether model_id, a local directory or a Hub repo, holds an adapter_config.json. Only a missing file means no: a gated repo, a bad token, a repo that does not exist or no network raises (peft's own check folds all of those into "can't find adapter_config.json"). Offline with the file not cached counts as missing."""
    if os.path.isdir(model_id):
        return os.path.isfile(os.path.join(model_id, "adapter_config.json"))
    try:
        hf_hub_download(model_id, "adapter_config.json")
    except EntryNotFoundError:  # includes LocalEntryNotFoundError, the offline case
        return False
    return True

def load_hf_model(model_id: str, parent_model_id: str | None = None, dtype=t.bfloat16, device_map="auto", **from_pretrained_kwargs) -> AutoModelForCausalLM:
    """Load a Hub model. If `model_id` is a peft adapter repo (is_adapter_repo), load its base (or `parent_model_id`) and merge the adapter in memory.
    Extra kwargs (quantization_config, max_memory, revision, ...) go to AutoModelForCausalLM.from_pretrained, the base model's on an adapter repo. In a pre-quantized checkpoint whose keys are renamed on load (DeepSeek-V4's all are), tensors keep their stored dtype whatever `dtype` says."""
    if not is_adapter_repo(model_id):
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map, **from_pretrained_kwargs)
    base_id = parent_model_id or PeftConfig.from_pretrained(model_id).base_model_name_or_path
    print(f"{gray}adapter repo: loading base '{orange}{base_id}{gray}' + adapter '{orange}{model_id}{gray}'{endc}")
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype, device_map=device_map, **from_pretrained_kwargs)
    return PeftModel.from_pretrained(base, model_id).merge_and_unload()

def boot_bridge(hf_model, dtype=None) -> TransformerBridge:
    """TransformerBridge around an HF model already in memory, in eval mode with grads off, left on the device from_pretrained put it (a device_map split over several GPUs stays split; TransformerLens refuses a map that mixes CPU and GPU). `dtype` is the Bridge's, the model's first parameter's when None. Load any peft adapters (load_adapters) before this: booting replaces the model's linear modules with the Bridge's wrappers, which peft cannot inject into, while a LoRA module loaded first is wrapped like any other and stays toggleable."""
    model = TransformerBridge.boot_transformers(hf_model.config.name_or_path, hf_model=hf_model, dtype=next(hf_model.parameters()).dtype if dtype is None else dtype, device=next(hf_model.parameters()).device)
    model.eval()
    model.requires_grad_(False)
    return model

def load_bridge(model_id: str, parent_model_id: str | None = None, dtype=t.bfloat16, device_map="auto", **from_pretrained_kwargs) -> TransformerBridge:
    """boot_bridge around load_hf_model's result. Extra kwargs go to from_pretrained (see load_hf_model)."""
    return boot_bridge(load_hf_model(model_id, parent_model_id, dtype, device_map, **from_pretrained_kwargs), dtype)

def adapter_spec(spec: str) -> tuple[str, dict]:
    """`repo`, `repo:subdir` (an adapter in a subdirectory of a Hub model repo) or a local directory, as (model_id, hub kwargs) for peft's loaders and hf_hub_download."""
    if os.path.isdir(spec):
        return spec, {}
    repo, _, sub = spec.partition(":")
    return repo, {"subfolder": sub} if sub else {}

def load_adapters(model, adapters: dict[str, str], **kwargs) -> PeftModel:
    """Named peft adapters on one model, kept toggleable rather than merged. `adapters` maps a name to `repo`, `repo:subdir` or a local directory (see adapter_spec). `model` is a raw HF model, loaded with load_hf_model and not yet booted into a Bridge (boot_bridge comes after, and then runs through the active adapters, since peft wraps the projection modules in place and the Bridge wraps those) or the PeftModel of an earlier call, which gets the new names; a name already loaded raises. Extra kwargs go to peft's load_adapter: `revision`, `token`, `autocast_adapter_dtype` (peft's default True casts bf16 adapter weights to float32).
    Afterwards, with peft's own API: `model.set_adapter(name)` activates one; `model.base_model.set_adapter([names])` a stack, whose LoRA deltas add, so an adapter trained on a merged base plus that base's adapter is exact; `with model.disable_adapter():` runs the base model, which is how activations are captured; `model.base_model.merge_adapter()` / `unmerge_adapter()` fold the active adapters into the weights for faster generation, where a merged adapter hides every other active one and each merge-unmerge cycle rounds the weights once in bf16. peft sets requires_grad False on the base model's parameters.
    Raises when a saved tensor lands on no module or a LoRA module gets no tensor: an adapter that silently matches nothing samples plausible base-model text, which every checkpoint card warns about."""
    if any(isinstance(m, GeneralizedComponent) for m in model.modules()):
        raise TypeError("the model is already wrapped by a TransformerBridge, whose component wrappers peft cannot inject into; load adapters on the HF model first, then boot_bridge it")
    for name, spec in adapters.items():
        model_id, hub = adapter_spec(spec)
        if isinstance(model, PeftModel) and name in model.peft_config:
            raise ValueError(f"adapter {name!r} is already loaded")
        if not isinstance(model, PeftModel):
            cfg = PeftConfig.from_pretrained(model_id, **hub)
            cfg.inference_mode = True
            model = PeftModel(model, cfg, adapter_name=name, **{k: kwargs[k] for k in ("autocast_adapter_dtype", "low_cpu_mem_usage") if k in kwargs})
        result = model.load_adapter(model_id, name, **hub, **kwargs)
        missing = [k for k in result.missing_keys if name in k.split(".")]  # peft's own filter matches the name as a substring, so 'b' matches every 'base_model' key
        if result.unexpected_keys or missing:
            raise ValueError(f"adapter {name!r} ({spec}) does not match the model: {len(result.unexpected_keys)} saved tensors with no module, {len(missing)} LoRA modules with no tensor, e.g. {[*result.unexpected_keys, *missing][:3]}")
    model.eval()
    return model

def check_active(model, names) -> None:
    """Raise unless `names` are exactly the active adapters of the PeftModel and adapters are enabled, so a metamodel read with the wrong adapter, none, or inside `disable_adapter()` fails instead of returning plausible base-model text. A model that is not a PeftModel (merged and unloaded) passes; the read functions never change adapter state themselves."""
    if not isinstance(model, PeftModel):
        return
    if any(m.disable_adapters for m in model.modules() if isinstance(m, BaseTunerLayer)):
        raise RuntimeError(f"adapters are disabled (a disable_adapter() block?); {sorted(names)} must be active to read")
    if set(model.active_adapters) != set(names):
        raise RuntimeError(f"active adapters {model.active_adapters} are not {sorted(names)}; activate them with model.set_adapter(name) or model.base_model.set_adapter([names])")
