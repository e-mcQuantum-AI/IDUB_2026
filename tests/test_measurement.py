import numpy as np
import pytest

from src.physics import WignerMeasurement, HusimiMeasurement, CoherentState, FockState


class TestWignerMeasurement:
    def test_output_shape(self, cutoff):
        """Wigner function output matrix must match the grid resolution dimensions."""
        resolution = 48
        state = CoherentState(alpha=1.0, cutoff=cutoff)
        meas = WignerMeasurement(x_max=4.0, resolution=resolution)

        W = meas.measure(state.density_matrix())
        assert W.shape == (resolution, resolution)

    def test_normalization(self, cutoff):
        """The integral of the Wigner function over the entire phase space must equal 1."""
        state = CoherentState(alpha=1.5, cutoff=cutoff)
        meas = WignerMeasurement(x_max=6.0, resolution=128)
        W = meas.measure(state.density_matrix())

        dx = meas.xvec[1] - meas.xvec[0]
        dp = dx
        integral = np.sum(W) * dx * dp

        assert abs(integral - 1.0) < 1e-3

    def test_vacuum_analytical_match(self, cutoff):
        """Wigner function of vacuum |0> must perfectly match the analytical Gaussian profile."""
        state = FockState(n=0, cutoff=cutoff)
        meas = WignerMeasurement(x_max=3.0, resolution=40)
        W_computed = meas.measure(state.density_matrix())

        X, P = np.meshgrid(meas.xvec, meas.xvec)
        W_analytical = (1.0 / np.pi) * np.exp(-(X ** 2 + P ** 2))

        assert np.max(np.abs(W_computed - W_analytical)) < 1e-6

    def test_fock_state_has_negativity(self, cutoff):
        """Kluczowy test: Funkcja Wignera stanu |1> MUSI przybierać wartości ujemne w centrum.

        Dla stanu Focka |1>, analityczna wartość w środku układu (x=0, p=0)
        wynosi dokładnie -1/pi ≈ -0.3183.
        """
        state = FockState(n=1, cutoff=cutoff)
        meas = WignerMeasurement(x_max=3.0, resolution=51)
        W = meas.measure(state.density_matrix())

        mid_idx = 51 // 2
        w_center = W[mid_idx, mid_idx]

        assert w_center < 0
        assert abs(w_center - (-1.0 / np.pi)) < 1e-4

class TestHusimiMeasurement:
    def test_always_non_negative(self, cutoff):
        """Husimi Q-function must be strictly non-negative everywhere, even for non-classical states."""
        state = FockState(n=1, cutoff=cutoff)
        meas = HusimiMeasurement(x_max=4.0, resolution=32)
        Q = meas.measure(state.density_matrix())

        assert np.all(Q >= -1e-14)

    def test_normalization(self, cutoff):
        """The integral of the Husimi Q-function over the entire phase space must equal 1."""
        state = CoherentState(alpha=1.0, cutoff=cutoff)
        meas = HusimiMeasurement(x_max=6.0, resolution=100)
        Q = meas.measure(state.density_matrix())

        dx = meas.xvec[1] - meas.xvec[0]
        dp = dx
        integral = np.sum(Q) * (0.5 * dx * dp)
        assert abs(integral - 1.0) < 1e-3

    def test_vacuum_analytical_match(self, cutoff):
        """Husimi Q-function of vacuum |0> must perfectly match the analytical Gaussian profile."""
        state = FockState(n=0, cutoff=cutoff)
        meas = HusimiMeasurement(x_max=3.0, resolution=40)
        Q_computed = meas.measure(state.density_matrix())

        X, P = np.meshgrid(meas.xvec, meas.xvec, indexing="ij")
        Q_analytical = (1.0 / np.pi) * np.exp(-0.5 * (X ** 2 + P ** 2))

        assert np.max(np.abs(Q_computed - Q_analytical)) < 1e-10

    def test_invalid_parameters(self):
        """Constructor must raise ValueError for unphysical grid settings."""
        with pytest.raises(ValueError):
            HusimiMeasurement(x_max=-2.0)
        with pytest.raises(ValueError):
            HusimiMeasurement(resolution=0)
