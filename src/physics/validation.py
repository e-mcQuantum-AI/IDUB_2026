"""Quantum state validity checks.

This module provides utility functions for verifying physical validity
of quantum density matrices.
"""

import numpy as np
from qutip import Qobj


def is_physical(rho: Qobj, atol: float = 1e-6) -> bool:
    """Check whether a quantum state is physically valid.

    A density matrix is considered physical if it satisfies:

    1. Hermiticity
    2. Unit trace
    3. Positive semi-definiteness

    Args:
        rho: Quantum object representing a density matrix.
        atol: Absolute tolerance used when checking numerical precision errors.

    Returns:
        bool: True if the state is physically valid, False otherwise.
    """
    # Hermiticity check
    if not rho.isherm:
        return False

    # Trace normalization
    if not np.isclose(rho.tr(), 1.0):
        return False

    # Positive semi-definiteness check
    eigvals = rho.eigenenergies()
    if np.any(eigvals < -atol):
        return False

    return True
