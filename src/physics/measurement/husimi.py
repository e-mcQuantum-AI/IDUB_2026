"""Husimi Q-function measurement module.

This module implements the Husimi Q-function phase-space measurement
for quantum optical states using an optimized, vectorized approach.
"""

import numpy as np
from qutip import Qobj


class HusimiMeasurement:
    """Husimi Q-function phase-space measurement.

    Computes the Husimi Q-function, defined as:

        Q(α) = 1/π * ⟨α|ρ|α⟩

    where |α⟩ is a coherent state and ρ is the density matrix of the state.
    The Husimi Q-function is a quasi-probability distribution that is
    always non-negative and represents a true probability distribution for
    simultaneous heterodyne measurements of the quadratures.

    This implementation uses a highly optimized, vectorized recurrence relation
    to construct the coherent state overlap matrix, avoiding slow nested loops
    and achieving O(cutoff * resolution²) performance.

    Args:
        x_max (float): Maximum value for both the X and P quadratures.
            The grid will span from -x_max to x_max. Defaults to 5.0.
        resolution (int): Number of grid points along each axis. Defaults to 64.
    """

    def __init__(self, x_max: float = 5.0, resolution: int = 64) -> None:
        """Initialize the Husimi Q-function measurement grid."""
        if x_max <= 0:
            raise ValueError("x_max must be positive.")
        if resolution <= 0:
            raise ValueError("resolution must be a positive integer.")

        self.x_max = x_max
        self.resolution = resolution
        self.xvec = np.linspace(-x_max, x_max, resolution)

    def measure(self, rho: Qobj) -> np.ndarray:
        """Compute the Husimi Q-function for the given density matrix.

        Args:
            rho (Qobj): Input quantum state (must be a density matrix / operator).

        Returns:
            np.ndarray: 2D array of shape (resolution, resolution) containing
                the values of Q(α), indexed such that Q[i, j] corresponds to
                xvec[i] (X quadrature) and xvec[j] (P quadrature).

        Raises:
            ValueError: If the input is not a valid density matrix operator.
        """
        if not rho.isoper:
            raise ValueError("Expected a density matrix (Qobj operator).")

        cutoff = rho.shape[0]

        X, P = np.meshgrid(self.xvec, self.xvec, indexing="ij")
        alpha = (X + 1j * P) / np.sqrt(2)
        alpha_flat = alpha.flatten()
        num_points = len(alpha_flat)

        # C[k, n] = <alpha_k | n> = exp(-|alpha|²/2) * (alpha*)^n / sqrt(n!)
        C = np.zeros((num_points, cutoff), dtype=complex)

        C[:, 0] = np.exp(-0.5 * np.abs(alpha_flat) ** 2)

        # <alpha | n> = <alpha | n-1> * alpha* / sqrt(n)
        alpha_conj = np.conj(alpha_flat)
        for n in range(1, cutoff):
            C[:, n] = C[:, n - 1] * alpha_conj / np.sqrt(n)

        rho_dense = rho.full()
        C_rho = C @ rho_dense
        Q_flat = np.real(np.sum(C_rho * np.conj(C), axis=1)) / np.pi

        return Q_flat.reshape((self.resolution, self.resolution))
