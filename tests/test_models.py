import pytest
import torch as t

from mechtools import models

class FakeHF:
    def __init__(self, name): self.name = name

class FakePeft:
    def __init__(self, base): self.base = base
    def merge_and_unload(self): return ("merged", self.base)

@pytest.fixture
def fakes(monkeypatch):
    """Adapter detection and loading with the Hub replaced: an id containing 'adapter' has a PeftConfig with base 'base/model', any other raises the ValueError peft raises for a plain checkpoint."""
    calls = []
    monkeypatch.setattr(models.AutoModelForCausalLM, "from_pretrained", lambda name, **kw: calls.append(("hf", name, kw)) or FakeHF(name))
    monkeypatch.setattr(models.PeftModel, "from_pretrained", lambda base, name: calls.append(("peft", name)) or FakePeft(base))
    def peft_config(name):
        if "adapter" not in name:
            raise ValueError("Can't find 'adapter_config.json'")
        return type("Cfg", (), {"base_model_name_or_path": "base/model"})()
    monkeypatch.setattr(models.PeftConfig, "from_pretrained", peft_config)
    return calls

def test_load_hf_model_plain(fakes):
    m = models.load_hf_model("plain/model", dtype=t.float16, device_map="cpu")
    assert m.name == "plain/model" and fakes == [("hf", "plain/model", {"dtype": t.float16, "device_map": "cpu"})]

def test_load_hf_model_adapter(fakes, capsys):
    m = models.load_hf_model("org/adapter")
    assert m == ("merged", fakes[0][2] and m[1]) and m[1].name == "base/model"
    assert fakes == [("hf", "base/model", {"dtype": t.bfloat16, "device_map": "auto"}), ("peft", "org/adapter")] and "adapter repo" in capsys.readouterr().out
    fakes.clear()
    m = models.load_hf_model("org/adapter", parent_model_id="other/base")  # the base can be overridden
    assert m[1].name == "other/base" and fakes[0][1] == "other/base"

@pytest.mark.hf
def test_load_bridge(tiny_bridge):
    model = tiny_bridge
    assert not model.training and not any(p.requires_grad for p in model.parameters()) and model.tokenizer is not None
    assert next(model.parameters()).dtype == t.float32 and model.generation_config.eos_token_id is not None
    ids = t.tensor([model.tokenizer.encode("Hello there")])
    logits = model(ids)
    assert logits.shape == (1, ids.shape[1], model.cfg.d_vocab)
