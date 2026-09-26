import builtins
import importlib
import inspect
import os
import pathlib
import random
import re
import subprocess
import sys

import numpy as np
import pytest
import torch as t

import mechtools
from mechtools import pbar, set_seed
from mechtools.colors import cyan

MODULES = ["colors", "hooks", "lens", "models", "openrouter", "plots", "resample", "sampling", "stats", "tables", "tokens"]
README = pathlib.Path(mechtools.__file__).parents[2] / "README.md"

def public(mod) -> dict:
    return {name: getattr(mod, name) for name in dir(mod) if not name.startswith("_")}

def test_star_import_has_no_collisions():
    """`from mechtools import *` re-exports every module's public names; two modules defining different objects under one name would silently shadow each other."""
    owners = {}
    for m in MODULES:
        for name, obj in public(importlib.import_module(f"mechtools.{m}")).items():
            if name in owners and owners[name][1] is not obj:
                pytest.fail(f"{name} is a different object in mechtools.{owners[name][0]} and mechtools.{m}")
            owners.setdefault(name, (m, obj))
            assert getattr(mechtools, name) is obj, name
    assert {"chat", "complete", "gather_bar", "Resampler", "probe", "show_logits", "imshow", "line", "kmeans", "wilson", "to_ids", "get_assistant_mask", "sample_rolling", "load_bridge", "add_bias_hook", "top_toks_table", "tec", "set_seed", "pbar", "cyan"} <= set(dir(mechtools))

@pytest.mark.skipif(not README.exists(), reason="README.md is only next to an editable install")
def test_readme_layout_names_exist():
    """Every backticked identifier in the README's Layout table names something in the package (a function, class, method or parameter), so renames update the docs."""
    layout = README.read_text().split("## Layout")[1].split("\n## ")[0]
    prose = {"HookedTransformer", "TransformerBridge", "text", "finish_reason", "prompt_tokens", "raw", "judge_error", "reasoning_content", "error"}  # classes from other packages, fields of flat records and messages, a finish_reason value
    for row in re.findall(r"^\| `(\w+)` \| (.*) \|$", layout, flags=re.M):
        mod = importlib.import_module(f"mechtools.{row[0]}")
        names = public(mod)
        known = set(names) | set(dir(builtins)) | prose
        for obj in names.values():
            for f in ([obj] + list(vars(obj).values()) if inspect.isclass(obj) else [obj]):  # a class, its methods and their parameters
                known |= set(vars(obj)) if inspect.isclass(obj) else set()
                if callable(f):
                    try:
                        known |= set(inspect.signature(f).parameters)
                    except (TypeError, ValueError):
                        pass
        missing = {tok for tok in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", row[1]) if tok not in known}
        assert not missing, f"README row for {row[0]} names {sorted(missing)}, which do not exist there"

def test_prelude_helpers():
    set_seed(3)
    a = (t.rand(2).tolist(), np.random.rand(), random.random())
    set_seed(3)
    assert a == (t.rand(2).tolist(), np.random.rand(), random.random())
    bar = pbar(range(3), desc="work")
    assert bar.desc.startswith(cyan + "work") and bar.ncols == 120 and bar.ascii == " >="
    bar.close()

def test_dotenv_is_loaded_from_the_cwd_on_import(tmp_path):
    """Importing mechtools loads the first .env found walking up from the current directory, not from the package's own directory."""
    (tmp_path / ".env").write_text("MECHTOOLS_TEST_ENV=from-cwd\n")
    env = {k: v for k, v in os.environ.items() if k != "MECHTOOLS_TEST_ENV"} | {"HF_HUB_OFFLINE": "1"}
    out = subprocess.run([sys.executable, "-c", "import os, mechtools; print('value:', os.environ.get('MECHTOOLS_TEST_ENV'))"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0 and "value: from-cwd" in out.stdout, out.stderr[-800:]
