import numpy as np

def is_physical(rho):
    if not rho.isherm:
        return False

    if not np.isclose(rho.tr(), 1.0):
        return False

    eigvals = rho.eigenenergies()
    if np.any(eigvals < -1e-10):
        return False

    return True
