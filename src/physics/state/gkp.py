import numpy as np
from qutip import displace, basis, Qobj
from .base import QuantumState

class GKPState(QuantumState):
    """ Approximate Gottesman-Kitaev-Preskill(GKP) state.

     GKP state encodes a logical qubit in a harmonic oscillator,
     protecting against small displacement errors in position (q)
     and momentum (p). The ideal state is an infinite comb of
     Dirac deltas spaced by 2*sqrt(pi). This implementation uses
     a finite-energy approximation with a Gaussian envelope of
     width delta:

        |GKP> ~ sum_s exp (-2*pi*s**2 / delta **2) D (s*sqrt(2*pi)) |0 >

     Reference :
        D . Gottesman , A . Kitaev , J . Preskill ,
        "Encoding a qubit in an oscillator " ,
        Phys . Rev . A 64 , 012310 (2001) .
        https :// doi . org /10.1103/ PhysRevA .64.012310

     Args :
        cutoff : Hilbert space dimension.
        delta : Envelope width (smaller = more ideal).
        grid_size : Number of peaks on each side.
    """

    def __init__(self, cutoff: int, delta: float = 0.3, grid_size: int = 5) -> None:
        self.cutoff = cutoff
        self.delta = delta
        self.grid_size = grid_size

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
