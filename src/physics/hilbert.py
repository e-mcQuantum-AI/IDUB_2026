from qutip import destroy, create, qeye
from dataclasses import dataclass
from qutip.core.cy.qobjevo import Qobj

@dataclass
class HilbertSpace:
    cutoff: int

    def a(self) -> Qobj:
        return destroy(self.cutoff)

    def adag(self) -> Qobj:
        return create(self.cutoff)

    def identity(self) -> Qobj:
        return qeye(self.cutoff)
