from tqdm import tqdm

from mechtools.colors import cyan

def pbar(iterable=None, desc: str = "", color: str = cyan, ncols: int = 120, **kwargs) -> tqdm:
    """tqdm with the house style: colored description, fixed width, ascii ' >=' fill. kwargs go to tqdm, e.g. disable=True to hide the bar, which is what the samplers' quiet flags pass."""
    return tqdm(iterable, desc=color + desc, ncols=ncols, ascii=" >=", **kwargs)
