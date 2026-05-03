"""Gottesman-Kitaev-Preskill (GKP) quantum state.

This module implements an approximate finite-energy GKP state,
used in bosonic quantum error correction.
"""

import numpy as np
from qutip import displace, basis, Qobj
from .base import QuantumState


class GKPState(QuantumState):
    """Approximate Gottesman-Kitaev-Preskill (GKP) state.

    The GKP state encodes a logical qubit in a harmonic oscillator,
    providing protection against small displacement errors in position (q)
    and momentum (p).

    The ideal state is an infinite comb of Dirac delta peaks spaced by:

        2√π

    This implementation uses a finite-energy approximation with a Gaussian
    envelope of width δ:

        |GKP⟩ ∝ ∑_s exp(-2π s² / δ²) D(s√(2π)) |0⟩

    where D(·) is the displacement operator.

    References:
        Gottesman, D., Kitaev, A., Preskill, J. (2001).
        "Encoding a qubit in an oscillator".
        Phys. Rev. A 64, 012310.
        https://doi.org/10.1103/PhysRevA.64.012310

    Args:
        cutoff (int): Hilbert space dimension.
        delta (float): Envelope width (smaller → closer to ideal GKP state).
        grid_size (int): Number of peaks on each side of the origin.

    Raises:
        ValueError: If cutoff is non-positive or delta is non-positive.
    """

    def __init__(self, cutoff: int, delta: float = 0.3, grid_size: int = 5) -> None:
        """Initialize a GKP state.

        Args:
            cutoff (int): Hilbert space dimension.
            delta (float): Envelope width.
            grid_size (int): Number of grid peaks per side.
        """
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        if delta <= 0:
            raise ValueError("delta must be positive.")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive.")

        self.cutoff = cutoff
        self.delta = delta
        self.grid_size = grid_size

    def ket(self) -> Qobj:
        """Construct the approximate GKP state.

        Builds a finite superposition of displaced vacuum states weighted
        by a Gaussian envelope.

        Returns:
            Qobj: Normalized GKP ket vector.
        """
        psi: Qobj = Qobj([[0]] * self.cutoff)
        vacuum: Qobj = basis(self.cutoff, 0)

        for s in range(-self.grid_size, self.grid_size + 1):

            q_shift = 2 * s * np.sqrt(np.pi)

            # displacement operator
            D: Qobj = displace(self.cutoff, q_shift / np.sqrt(2))

            # Gaussian envelope
            weight = np.exp(-(2 * s * np.sqrt(np.pi)) ** 2 / (2 * self.delta ** 2))

            psi += weight * D * vacuum

        return psi.unit()
