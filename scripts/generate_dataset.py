from src.physics.measurement.wigner import WignerMeasurement
from src.physics.noise.loss import LossChannel
from src.physics.measurement.pipeline import MeasurementPipeline
from src.dataset.generator import DatasetGenerator
from src.dataset.state_sampler import simple_sampler
import numpy as np


if __name__ == '__main__':
    cutoff = 40

    measurement = WignerMeasurement()
    noise = LossChannel(cutoff=cutoff, gamma=0.1)

    pipeline = MeasurementPipeline(measurement, noise)

    generator = DatasetGenerator(
        lambda: simple_sampler(cutoff),
        pipeline
    )

    X, y = generator.generate(1000)

    np.savez("dataset_v0.npz", X=X, y=y)
