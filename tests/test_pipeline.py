import pytest
import numpy as np
from src.physics import FockState, WignerMeasurement, MeasurementPipeline, LossChannel, CoherentState, CatState, \
    VacuumState, GKPState, BinomialState, ThermalState


class TestMeasurementPipeline:
    def test_clean_measurement(self, cutoff):
        """Pipeline bez szumu produkuje tablicę 2D (obraz Wignera)."""
        state = FockState(n=2, cutoff=cutoff)
        wm = WignerMeasurement(x_max=5, resolution=64)
        pipeline = MeasurementPipeline(measurement=wm)
        result = pipeline.run(state)
        assert result.shape == (64, 64)
        assert np.isfinite(result).all()

    def test_noisy_measurement(self, cutoff):
        """Pipeline z szumem powinien zmienić obraz Wignera (szum = widoczna różnica)."""
        state = FockState(n=3, cutoff=cutoff)
        wm = WignerMeasurement(x_max=5, resolution=64)
        noise = LossChannel(cutoff=cutoff, gamma=0.5)
        pipeline_clean = MeasurementPipeline(measurement=wm)
        pipeline_noisy = MeasurementPipeline(measurement=wm, noise=noise)
        w_clean = pipeline_clean.run(state)
        w_noisy = pipeline_noisy.run(state)
        # Noisy should differ from clean
        assert not np.allclose(w_clean, w_noisy)

    def test_all_states_produce_output(self, cutoff):
        """Wszystkie typy stanów powinny dać poprawny obraz Wignera (brak NaN/Inf)."""
        wm = WignerMeasurement(x_max=5, resolution=32)
        pipeline = MeasurementPipeline(measurement=wm)
        states = [
            FockState(n=1, cutoff=cutoff),
            CoherentState(alpha=1.0, cutoff=cutoff),
            CatState(alpha=2.0, cutoff=cutoff),
            VacuumState(cutoff=cutoff),
            GKPState(cutoff=cutoff, delta=0.4),
            BinomialState(N=3, p=0.5, cutoff=cutoff),
        ]
        for state in states:
            result = pipeline.run(state)
            assert result.shape == (32, 32)
            assert np.isfinite(result).all()

    def test_thermal_through_pipeline(self, cutoff):
        """Stan mieszany (termiczny) też powinien działać w pipeline.

        Termiczny to stan mieszany (nie ma ketu), więc pipeline musi
        poprawnie obsłużyć ścieżkę density_matrix() → pomiar.
        """
        state = ThermalState(n_th=1.0, cutoff=cutoff)
        wm = WignerMeasurement(x_max=5, resolution=32)
        pipeline = MeasurementPipeline(measurement=wm)
        result = pipeline.run(state)
        assert result.shape == (32, 32)
