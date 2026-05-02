from qutip import Qobj


class MeasurementPipeline:

    def __init__(self, measurement, noise=None) -> None:
        self.measurement = measurement
        self.noise = noise

    def run(self, state) -> Qobj:
        rho: Qobj = state.density_matrix()

        if self.noise:
            rho = self.noise.apply(rho)

        return self.measurement.measure(rho)
