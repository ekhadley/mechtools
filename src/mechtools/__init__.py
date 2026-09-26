import random

import IPython
import numpy as np
import torch as t
from dotenv import find_dotenv, load_dotenv

from mechtools.bars import *
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

load_dotenv(find_dotenv(usecwd=True))  # the first .env found walking up from the cwd: the project's, when run from its directory

def tec(): t.cuda.empty_cache()

def set_seed(seed: int) -> None:
    t.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
