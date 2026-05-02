from qutip import Qobj

from .base import QuantumChannel


class MixtureChannel(QuantumChannel):

    def __init__(self, p: float, rho_other: Qobj):
        self.p = p
        self.rho_other = rho_other

    def apply(self, rho: Qobj) -> Qobj:
        return self.p * rho + (1 - self.p) * self.rho_other
