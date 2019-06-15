import pytest

# @pytest.mark.skip(reason="Too slow")

def test_cartesian_production():
    import numpy as np
    from src.models.low_rank.fourier_features import cartesian_product

    a1 = np.array([1,2,3])
    a2 = np.array([1,2,3])
    a3 = np.array([1,2,3])
    p = cartesian_product(*(a1, a2, a3))
    assert p.shape == (len(a1) * len(a2) * len(a3), 3), "The cartesian product does not have the right shape"
