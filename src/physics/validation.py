"""Quantum state validity checks.

This module provides utility functions for verifying physical validity
of quantum density matrices.
"""

import numpy as np


def is_physical(rho) -> bool:
    """Check whether a quantum state is physically valid.

    A density matrix is considered physical if it satisfies:

    1. Hermiticity
    2. Unit trace
    3. Positive semi-definiteness

    Args:
        rho: Quantum object representing a density matrix.

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
    if np.any(eigvals < -1e-10):
        return False

    return True
