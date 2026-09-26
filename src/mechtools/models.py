import os

import torch as t
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from peft import PeftConfig, PeftModel
from transformer_lens.model_bridge import TransformerBridge
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

def load_hf_model(model_id: str, parent_model_id: str | None = None, dtype=t.bfloat16, device_map="auto") -> AutoModelForCausalLM:
    """Load a Hub model. If `model_id` is a peft adapter repo (is_adapter_repo), load its base (or `parent_model_id`) and merge the adapter in memory."""
    if not is_adapter_repo(model_id):
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)
    base_id = parent_model_id or PeftConfig.from_pretrained(model_id).base_model_name_or_path
    print(f"{gray}adapter repo: loading base '{orange}{base_id}{gray}' + adapter '{orange}{model_id}{gray}'{endc}")
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype, device_map=device_map)
    return PeftModel.from_pretrained(base, model_id).merge_and_unload()

def load_bridge(model_id: str, parent_model_id: str | None = None, dtype=t.bfloat16, device_map="auto") -> TransformerBridge:
    """TransformerBridge around load_hf_model's result, in eval mode with grads off."""
    hf_model = load_hf_model(model_id, parent_model_id, dtype, device_map)
    model = TransformerBridge.boot_transformers(hf_model.config.name_or_path, hf_model=hf_model, dtype=dtype, device=next(hf_model.parameters()).device)
    model.eval()
    model.requires_grad_(False)
    return model
