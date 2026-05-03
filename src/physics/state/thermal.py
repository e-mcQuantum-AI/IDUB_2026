"""Thermal quantum state.

This module defines a thermal (Gibbs) state of a bosonic mode,
which is a mixed quantum state rather than a pure state.
"""

from qutip import thermal_dm, Qobj
from .base import QuantumState


class ThermalState(QuantumState):
    """Thermal (Gibbs) quantum state.

    A thermal state is a mixed state describing a bosonic mode in
    equilibrium with a heat bath. It is not representable as a pure
    ket vector.

    The density matrix is given by:

        ρ = thermal_dm(N, n̄)

    where:
        N   — cutoff (Hilbert space dimension)
        n̄  — average thermal occupation number

    Args:
        n_th (float): Mean thermal occupation number.
        cutoff (int): Hilbert space dimension.

    Raises:
        ValueError: If n_th < 0 or cutoff is non-positive.
    """

    def __init__(self, n_th: float, cutoff: int) -> None:
        """Initialize a thermal state.

        Args:
            n_th (float): Mean thermal photon number.
            cutoff (int): Hilbert space dimension.
        """
        if n_th < 0:
            raise ValueError("n_th must be non-negative.")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")

        self.n_th = n_th
        self.cutoff = cutoff

    def ket(self) -> Qobj:
        """Thermal states do not have a ket representation.

        Raises:
            NotImplementedError: Always, since thermal states are mixed.
        """
        raise NotImplementedError("Thermal state is a mixed state; no ket representation exists.")

    def density_matrix(self) -> Qobj:
        """Return the thermal density matrix.

        Returns:
            Qobj: Density matrix of the thermal state.
        """
        return thermal_dm(self.cutoff, self.n_th)
