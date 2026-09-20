import random

import IPython
import numpy as np
import torch as t
from dotenv import load_dotenv
from tqdm import tqdm

from mechtools.colors import *
from mechtools.hooks import *
from mechtools.lens import *
from mechtools.models import *
from mechtools.openrouter import *
from mechtools.plots import *
from mechtools.resample import *
from mechtools.sampling import *
from mechtools.stats import *
from mechtools.tables import *
from mechtools.tokens import *

IPYTHON = IPython.get_ipython()
if IPYTHON is not None:
    IPYTHON.run_line_magic("load_ext", "autoreload")
    IPYTHON.run_line_magic("autoreload", "2")

load_dotenv()

def tec(): t.cuda.empty_cache()

def set_seed(seed: int) -> None:
    t.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def pbar(iterable=None, desc: str = "", color: str = cyan, ncols: int = 120, **kwargs) -> tqdm:
    """tqdm with the house style: colored description, fixed width, ascii ' >=' fill."""
    return tqdm(iterable, desc=color + desc, ncols=ncols, ascii=" >=", **kwargs)
