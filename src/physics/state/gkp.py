import numpy as np
from qutip import displace, basis
from base import QuantumState

class GKPState(QuantumState):

    def __init__(self, cutoff: int, delta: float = 0.3, grid_size: int = 5):
        self.cutoff = cutoff
        self.delta = delta
        self.grid_size = grid_size  # liczba pików po obu stronach

    def ket(self):

        psi = 0
        vacuum = basis(self.cutoff, 0)

        for s in range(-self.grid_size, self.grid_size + 1):

            q_shift = 2 * s * np.sqrt(np.pi)

            # operator przesunięcia
            D = displace(self.cutoff, q_shift / np.sqrt(2))

            # envelope Gaussowski
            weight = np.exp(- (2 * s * np.sqrt(np.pi))**2 / (2 * self.delta**2))

            psi += weight * D * vacuum

        return psi.unit()
