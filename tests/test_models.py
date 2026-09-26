from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError, RemoteEntryNotFoundError

import mechtools.models as mm
from mechtools.models import is_adapter_repo, load_hf_model

def hub_error(cls, status: int, msg: str):
    return cls(msg, response=httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/x")))

def test_is_adapter_repo_local(tmp_path):
    assert not is_adapter_repo(str(tmp_path))
    (tmp_path / "adapter_config.json").write_text("{}")
    assert is_adapter_repo(str(tmp_path))

def test_is_adapter_repo_hub_only_treats_a_missing_file_as_no(monkeypatch):
    def download(repo_id, filename):
        assert filename == "adapter_config.json"
        if repo_id == "org/adapter": return "/cache/adapter_config.json"
        if repo_id == "org/base": raise hub_error(RemoteEntryNotFoundError, 404, "entry not found")
        if repo_id == "org/offline-base": raise LocalEntryNotFoundError("not in the cache and outgoing traffic is disabled")
        if repo_id == "org/gated": raise hub_error(GatedRepoError, 403, "gated repo")
        raise httpx.ConnectError("no network")
    monkeypatch.setattr(mm, "hf_hub_download", download)
    assert is_adapter_repo("org/adapter") and not is_adapter_repo("org/base") and not is_adapter_repo("org/offline-base")
    with pytest.raises(GatedRepoError):  # peft would have folded this into "can't find adapter_config.json"
        is_adapter_repo("org/gated")
    with pytest.raises(httpx.ConnectError):
        is_adapter_repo("org/nowhere")

def test_load_hf_model_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(mm, "is_adapter_repo", lambda mid: mid.endswith("-lora"))
    monkeypatch.setattr(mm.AutoModelForCausalLM, "from_pretrained", lambda mid, **kw: calls.append(("auto", mid)) or SimpleNamespace(id=mid))
    monkeypatch.setattr(mm.PeftConfig, "from_pretrained", lambda mid: calls.append(("cfg", mid)) or SimpleNamespace(base_model_name_or_path="org/base"))
    monkeypatch.setattr(mm.PeftModel, "from_pretrained", lambda base, mid: calls.append(("peft", base.id, mid)) or SimpleNamespace(merge_and_unload=lambda: SimpleNamespace(id=f"{base.id}+{mid}")))
    assert load_hf_model("org/base").id == "org/base" and calls == [("auto", "org/base")]
    calls.clear()
    assert load_hf_model("org/x-lora").id == "org/base+org/x-lora" and calls == [("cfg", "org/x-lora"), ("auto", "org/base"), ("peft", "org/base", "org/x-lora")]
    calls.clear()
    assert load_hf_model("org/x-lora", parent_model_id="org/other").id == "org/other+org/x-lora" and calls[0] == ("auto", "org/other")
