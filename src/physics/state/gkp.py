import numpy as np
from qutip import displace, basis, Qobj
from .base import QuantumState

class GKPState(QuantumState):

    def __init__(self, cutoff: int, delta: float = 0.3, grid_size: int = 5) -> None:
        self.cutoff = cutoff
        self.delta = delta
        self.grid_size = grid_size  # number of peaks on both sides

    def ket(self) -> Qobj:
        psi: Qobj = Qobj([[0]] * self.cutoff)
        vacuum: Qobj = basis(self.cutoff, 0)

        for s in range(-self.grid_size, self.grid_size + 1):

            q_shift = 2 * s * np.sqrt(np.pi)

            # displacement operator
            D: Qobj = displace(self.cutoff, q_shift / np.sqrt(2))

            # Gaussian envelope
            weight = np.exp(- (2 * s * np.sqrt(np.pi))**2 / (2 * self.delta**2))

            psi += weight * D * vacuum

        return psi.unit()
