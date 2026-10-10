import json
import re

import pytest
import torch as t

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # 1M parameters, 2 layers, d 16, Llama-2 tokenizer; boots a TransformerBridge in under a second

def widget_data(h: str) -> dict:
    """The JSON payload of a tabbed readout's html, {"tabs", "inner", "panes"} with the first pane's entry None, as its script parses it: what the readout tests check the data against. A widget of one pane has none."""
    return json.loads(re.search(r"<script type=application/json id='\w+'>(.*?)</script>", h, re.S).group(1))

def first_pane(h: str) -> str:
    """The html of the one pane rendered into a tabbed readout's html, the pane shown where scripts do not run."""
    return re.search(r"<div class='pane'>(.*?</div>)</div></div>(?:</div>|<script type=application/json)", h, re.S).group(1)

def load_tokenizer(name: str):
    """AutoTokenizer.from_pretrained, skipping the test when the tokenizer is not in the offline HF cache."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name)
    except OSError as e:
        pytest.skip(f"{name} is not in the local HF cache ({type(e).__name__})")

@pytest.fixture(scope="session")
def tiny_bridge():
    """TransformerBridge around TINY_MODEL on cpu in float32, for integration tests of sampling, hooks, readouts and loading. Skips when the model is not cached."""
    from mechtools.models import load_bridge
    try:
        return load_bridge(TINY_MODEL, dtype=t.float32, device_map="cpu")
    except OSError as e:
        pytest.skip(f"{TINY_MODEL} is not in the local HF cache, run `hf download {TINY_MODEL}` ({type(e).__name__})")

HF_FIXTURES = {"tok", "qwen", "tiny_bridge"}

def pytest_collection_modifyitems(items):
    """Every test that uses a fixture backed by the HF cache carries the hf marker, so `pytest -m "not hf"` runs the pure tests anywhere."""
    for item in items:
        if HF_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.hf)
