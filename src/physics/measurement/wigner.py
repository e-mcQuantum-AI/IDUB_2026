"""Wigner function measurement for quantum states.

This module provides a measurement class that computes the Wigner
quasi-probability distribution of a quantum state represented as
a density matrix.
"""

import numpy as np
from qutip import wigner, Qobj


class WignerMeasurement:
    """Wigner function measurement.

    Computes the Wigner quasi-probability distribution on a 2D phase-space
    grid defined over position (q) and momentum (p) axes.

    The Wigner function provides a full phase-space representation of a
    quantum state and can reveal non-classical features such as negativity.

    Args:
        x_max (float): Maximum absolute value of phase-space coordinates.
        resolution (int): Number of points per axis in the grid.
    """

    def __init__(self, x_max: float = 5.0, resolution: int = 64) -> None:
        """Initialize the Wigner measurement grid.

        Args:
            x_max (float): Maximum absolute coordinate value.
            resolution (int): Number of grid points per axis.
        """
        self.xvec = np.linspace(-x_max, x_max, resolution)

    def measure(self, rho: Qobj) -> np.ndarray:
        """Compute the Wigner function of a quantum state.

        Args:
            rho (Qobj): Density matrix of the quantum state.

        Returns:
            np.ndarray: 2D array representing the Wigner function values
                over the phase-space grid.
        """
        W = wigner(rho, self.xvec, self.xvec)
        return W
