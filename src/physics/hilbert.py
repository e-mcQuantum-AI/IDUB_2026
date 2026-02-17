from qutip import destroy, create, qeye
from dataclasses import dataclass

@dataclass
class HilbertSpace:
    cutoff: int

    def a(self):
        return destroy(self.cutoff)

    def adag(self):
        return create(self.cutoff)

    def identity(self):
        return qeye(self.cutoff)
