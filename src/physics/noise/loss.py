from qutip import destroy, mesolve, Qobj
import numpy as np
from .base import QuantumChannel


class LossChannel(QuantumChannel):
    """ Photon loss (amplitude damping) channel.

    This channel models energy dissipation in a bosonic mode due to
    photon loss, described by a Lindblad master equation with a single
    collapse operator proportional to the annihilation operator.

    The evolution of the density matrix ρ is given by:

        dρ/dt = γ ( a ρ a† - 1/2 {a† a, ρ} )

    where a is the annihilation operator and γ is the loss rate.
    This process corresponds physically to coupling the mode to a
    zero-temperature bath, leading to exponential decay of excitations.

    In this implementation, the channel is obtained by solving the
    master equation over a fixed time interval t ∈ [0, 1], so γ
    effectively sets the strength of the loss.

    Args:
        cutoff: Hilbert space dimension.
        gamma: Loss rate (decay strength).
    """

    def __init__(self, cutoff: int, gamma: float):
        self.cutoff = cutoff
        self.gamma = gamma

    def apply(self, rho: Qobj) -> Qobj:
        a: Qobj = destroy(self.cutoff)
        result = mesolve(
            H=0 * a.dag() * a,
            rho0=rho,
            tlist=[0, 1],
            c_ops=[np.sqrt(self.gamma) * a]
        )
        return result.states[-1]
