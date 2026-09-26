import numpy as np
import pytest

from mechtools.plots import hist, line

def test_names_are_not_consumed():
    names = ["a", "b"]
    fig = line([np.arange(3), np.arange(3) * 2], names=names, return_fig=True)
    assert [tr.name for tr in fig.data] == ["a", "b"] and names == ["a", "b"]
    fig = hist([np.random.rand(20), np.random.rand(20)], names=names, return_fig=True)
    assert [tr.name for tr in fig.data] == ["a", "b"] and names == ["a", "b"]
    with pytest.raises(AssertionError, match="1 names for 2 lines"):
        line([np.arange(3), np.arange(3)], names=["only one"], return_fig=True)
    with pytest.raises(AssertionError, match="1 names for 2 series"):
        hist([np.random.rand(20), np.random.rand(20)], names=["only one"], return_fig=True)
