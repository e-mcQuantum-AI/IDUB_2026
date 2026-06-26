import pytest
import numpy as np
from qutip import destroy, expect, Qobj

from src.physics import (
    # Stany (7)
    VacuumState, FockState, CoherentState, CatState,
    GKPState, BinomialState, ThermalState,

    # Kanały szumu (7)
    LossChannel, DephasingChannel, DepolarizingChannel,
    AmplificationChannel, GaussianNoiseChannel,
    ThermalNoiseChannel, MixtureChannel,

    # Pomiary (2)
    WignerMeasurement, HusimiMeasurement,

    # Potok
    MeasurementPipeline
)
from src.physics.validation import is_physical

STATES = {
    "Vacuum": lambda c: VacuumState(cutoff=c),
    "Fock": lambda c: FockState(n=1, cutoff=c),
    "Coherent": lambda c: CoherentState(alpha=1.2, cutoff=c),
    "Cat": lambda c: CatState(alpha=1.5, cutoff=c),
    "GKP": lambda c: GKPState(delta=0.4, cutoff=c),
    "Binomial": lambda c: BinomialState(N=2, p=0.5, cutoff=c),
    "Thermal": lambda c: ThermalState(n_th=0.5, cutoff=c),
}

CHANNELS = {
    "Loss": lambda c: LossChannel(cutoff=c, gamma=0.1),
    "Dephasing": lambda c: DephasingChannel(cutoff=c, gamma=0.1),
    "Depolarizing": lambda c: DepolarizingChannel(cutoff=c, p=0.1),
    "Amplification": lambda c: AmplificationChannel(cutoff=c, gamma=0.1),
    "Gaussian": lambda c: GaussianNoiseChannel(cutoff=c, gamma=0.1),
    "ThermalNoise": lambda c: ThermalNoiseChannel(cutoff=c, gamma=0.1, n_th=0.2),
    "Mixture": lambda c: MixtureChannel(
        p=0.8,
        rho_other=ThermalState(n_th=0.5, cutoff=c).density_matrix()
    ),
}

MEASUREMENTS = {
    "Wigner": lambda: WignerMeasurement(x_max=4, resolution=16),
    "Husimi": lambda: HusimiMeasurement(x_max=4, resolution=16),
}


class TestMeasurementPipeline:
    @pytest.fixture
    def fast_cutoff(self):
        """Bezpieczny, ale szybki wymiar przestrzeni dla 98 symulacji."""
        return 12

    # =====================================================================
    # 1. TEST MATRYCOWY: 98 KOMBINACJI (7 x 7 x 2)
    # =====================================================================
    @pytest.mark.parametrize("state_name", STATES.keys())
    @pytest.mark.parametrize("channel_name", CHANNELS.keys())
    @pytest.mark.parametrize("meas_name", MEASUREMENTS.keys())
    def test_pipeline_98_combinations(self, state_name, channel_name, meas_name, fast_cutoff):
        """
        Testuje pełen przepływ: Stan -> Szum -> {Wigner, Husimi} -> Tablica 2D.
        """
        state = STATES[state_name](fast_cutoff)
        channel = CHANNELS[channel_name](fast_cutoff)
        meas = MEASUREMENTS[meas_name]()

        pipeline = MeasurementPipeline(measurement=meas, noise=channel)

        rho_in = state.density_matrix()

        assert is_physical(rho_in), f"Stan {state_name} nie jest fizyczny!"

        result_2d = pipeline.run(state)

        assert isinstance(result_2d, np.ndarray), "Wynik musi być macierzą NumPy."
        assert result_2d.shape == (16, 16), f"Błędny wymiar wyjściowy dla {meas_name}."
        assert np.all(np.isfinite(result_2d)), f"Tablica zawiera NaN/Inf w {state_name}+{channel_name}."

        if meas_name == "Husimi":
            assert np.all(result_2d >= -1e-13), f"Husimi Q < 0 dla {state_name} po {channel_name}!"

    # =====================================================================
    # 2. TESTY SPECYFICZNYCH CECH FIZYCZNYCH Z CHECKLISTY
    # =====================================================================
    def test_loss_reduces_photons(self, fast_cutoff):
        """Test kanałów szumu: Redukcja fotonów (straty)."""
        state = FockState(n=3, cutoff=fast_cutoff)
        rho_in = state.density_matrix()

        channel = LossChannel(cutoff=fast_cutoff, gamma=0.3)
        rho_out = channel.apply(rho_in)

        a = destroy(fast_cutoff)
        n_hat = a.dag() * a
        assert expect(n_hat, rho_out) < expect(n_hat, rho_in)

    def test_dephasing_conserves_photons(self, fast_cutoff):
        """Test kanałów szumu: Zachowanie fotonów (defazowanie)."""
        state = FockState(n=2, cutoff=fast_cutoff)
        rho_in = state.density_matrix()

        channel = DephasingChannel(cutoff=fast_cutoff, gamma=0.5)
        rho_out = channel.apply(rho_in)

        a = destroy(fast_cutoff)
        n_hat = a.dag() * a
        assert abs(expect(n_hat, rho_out) - expect(n_hat, rho_in)) < 1e-6

    def test_husimi_normalization(self, fast_cutoff):
        """Testy Husimi Q: Normalizacja całki do 1."""
        state = CoherentState(alpha=1.0, cutoff=fast_cutoff)
        meas = HusimiMeasurement(x_max=5.0, resolution=40)

        pipeline = MeasurementPipeline(measurement=meas)
        Q_matrix = pipeline.run(state)

        dx = meas.xvec[1] - meas.xvec[0]
        integral = np.sum(Q_matrix) * (0.5 * dx * dx)

        assert abs(integral - 1.0) < 1e-3

    # =====================================================================
    # 3. TESTY Z ZACHOWANIEM STAREJ FUNKCJONALNOŚCI
    # =====================================================================
    def test_noisy_vs_clean_difference(self, fast_cutoff):
        """Pipeline z szumem powinien widocznie zmienić obraz względem czystego."""
        state = FockState(n=2, cutoff=fast_cutoff)
        wm = WignerMeasurement(x_max=4, resolution=32)
        noise = LossChannel(cutoff=fast_cutoff, gamma=0.5)

        clean_pipeline = MeasurementPipeline(measurement=wm)
        noisy_pipeline = MeasurementPipeline(measurement=wm, noise=noise)

        w_clean = clean_pipeline.run(state)
        w_noisy = noisy_pipeline.run(state)

        assert not np.allclose(w_clean, w_noisy)
