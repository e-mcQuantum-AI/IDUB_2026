import pytest

@pytest.fixture
def cutoff():
    return 32

@pytest.fixture
def xvec():
    import numpy as np
    return np.linspace(-5, 5, 64)
