"""Measurement pipeline for quantum states.

This module defines a simple pipeline that applies an optional noise
channel to a quantum state and then performs a measurement on the
resulting density matrix.
"""

from qutip import Qobj


class MeasurementPipeline:
    """Pipeline for applying noise and performing a quantum measurement.

    This class composes two stages:
    1. Optional noise channel applied to a density matrix.
    2. Measurement performed on the resulting state.

    The input state is expected to provide a ``density_matrix()`` method
    returning a ``qutip.Qobj``.

    Args:
        measurement: Measurement object with a ``measure(Qobj)`` method.
        noise: Optional noise channel with an ``apply(Qobj)`` method.
    """

    def __init__(self, measurement, noise=None) -> None:
        """Initialize the measurement pipeline.

        Args:
            measurement: Measurement object implementing ``measure(rho)``.
            noise: Optional noise channel implementing ``apply(rho)``.
        """
        self.measurement = measurement
        self.noise = noise

    def run(self, state) -> Qobj:
        """Execute the pipeline on a quantum state.

        The state is first converted to a density matrix. If a noise
        channel is provided, it is applied before performing the
        measurement.

        Args:
            state: Quantum state with a ``density_matrix() -> Qobj`` method.

        Returns:
            Qobj: Result of the measurement.

        Raises:
            AttributeError: If the input state does not implement
                ``density_matrix()``.
        """
        rho: Qobj = state.density_matrix()

        if self.noise:
            rho = self.noise.apply(rho)

        return self.measurement.measure(rho)
