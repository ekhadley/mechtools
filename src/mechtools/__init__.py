import random

import IPython
import numpy as np
import torch as t
from dotenv import load_dotenv

from mechtools.colors import *
from mechtools.hooks import *
from mechtools.models import *
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
