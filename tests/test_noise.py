import numpy as np
import pytest
from qutip import destroy, expect, qeye

from src.physics import (
    QuantumChannel,
    CoherentState,
    LossChannel,
    FockState,
    MixtureChannel,
    DephasingChannel,
    DepolarizingChannel,
    AmplificationChannel,
    GaussianNoiseChannel,
    ThermalNoiseChannel
)
from src.physics.validation import is_physical


class TestLossChannel:
    def test_preserves_physicality(self, cutoff):
        """Po zastosowaniu strat stan nadal musi być fizyczny.

        Nawet po utracie fotonów, macierz gęstości musi spełniać
        warunki fizyczności (hermitowskość, ślad=1, wartości własne ≥ 0).
        """
        state = CoherentState(alpha=2.0, cutoff=cutoff)
        channel = LossChannel(cutoff=cutoff, gamma=0.5)
        rho_out = channel.apply(state.density_matrix())
        assert is_physical(rho_out)

    def test_reduces_photon_number(self, cutoff):
        """Kanał strat powinien zmniejszyć średnią liczbę fotonów.

        Straty = fotony uciekają z układu → po stratach jest ich mniej.
        Porównujemy ⟨n̂⟩ przed i po zastosowaniu kanału.
        """
        state = FockState(n=5, cutoff=cutoff)
        channel = LossChannel(cutoff=cutoff, gamma=0.3)
        rho_in = state.density_matrix()
        rho_out = channel.apply(rho_in)
        a = destroy(cutoff)
        n_hat = a.dag() * a         # operator liczby fotonów
        assert expect(n_hat, rho_out) < expect(n_hat, rho_in)

    def test_zero_loss_preserves_state(self, cutoff):
        """Przy zerowych stratach (γ=0) stan nie powinien się zmienić.

        Test „zdrowego rozsądku" — brak szumu = brak zmian.
        """
        state = CoherentState(alpha=1.0, cutoff=cutoff)
        channel = LossChannel(cutoff=cutoff, gamma=0.0)
        rho_in = state.density_matrix()
        rho_out = channel.apply(rho_in)
        assert abs((rho_in - rho_out).norm()) < 1e-6

    def test_is_quantum_channel(self, cutoff):
        """LossChannel powinien być instancją QuantumChannel (poprawne dziedziczenie)."""
        channel = LossChannel(cutoff=cutoff, gamma=0.1)
        assert isinstance(channel, QuantumChannel)

    def test_invalid_gamma_raises(self, cutoff):
        """Kanał strat musi odrzucić ujemną lub większą od 1 wartość gamma."""
        with pytest.raises(ValueError):
            LossChannel(cutoff=cutoff, gamma=1.5)
        with pytest.raises(ValueError):
            LossChannel(cutoff=cutoff, gamma=-0.1)

class TestMixtureChannel:
    def test_preserves_physicality(self, cutoff):
        """Mieszanina dwóch stanów fizycznych musi być fizyczna."""
        s1 = FockState(n=0, cutoff=cutoff).density_matrix()
        s2 = FockState(n=1, cutoff=cutoff).density_matrix()
        channel = MixtureChannel(p=0.7, rho_other=s2)
        rho_out = channel.apply(s1)
        assert is_physical(rho_out)

    def test_p_one_returns_original(self, cutoff):
        """Przy p=1 mieszanina zwraca oryginalny stan (100% wagi na ρ, 0% na ρ_other)."""
        s1 = CoherentState(alpha=1.0, cutoff=cutoff).density_matrix()
        s2 = FockState(n=0, cutoff=cutoff).density_matrix()
        channel = MixtureChannel(p=1.0, rho_other=s2)
        rho_out = channel.apply(s1)
        assert abs((s1 - rho_out).norm()) < 1e-10

    def test_p_zero_returns_other(self, cutoff):
        """Przy p=0 mieszanina zwraca drugi stan (0% wagi na ρ, 100% na ρ_other)."""
        s1 = CoherentState(alpha=1.0, cutoff=cutoff).density_matrix()
        s2 = FockState(n=0, cutoff=cutoff).density_matrix()
        channel = MixtureChannel(p=0.0, rho_other=s2)
        rho_out = channel.apply(s1)
        assert abs((s2 - rho_out).norm()) < 1e-10

    def test_invalid_p_raises(self, cutoff):
        """Mieszanina musi odrzucić prawdopodobieństwo spoza zakresu [0, 1]."""
        s2 = FockState(n=0, cutoff=cutoff).density_matrix()
        with pytest.raises(ValueError):
            MixtureChannel(p=1.2, rho_other=s2)
        with pytest.raises(ValueError):
            MixtureChannel(p=-0.5, rho_other=s2)


class TestDephasingChannel:
    def test_preserves_physicality(self, cutoff):
        """After dephasing, the state must remain physical (is_physical is True)."""
        state = CoherentState(alpha=1.5, cutoff=cutoff)
        channel = DephasingChannel(cutoff=cutoff, gamma=0.2)
        rho_out = channel.apply(state.density_matrix())
        assert is_physical(rho_out)

    def test_conserves_photon_number(self, cutoff):
        """Dephasing must conserve the mean photon number, as [n̂, n̂] = 0."""
        state = CoherentState(alpha=2.0, cutoff=cutoff)
        rho_in = state.density_matrix()
        channel = DephasingChannel(cutoff=cutoff, gamma=0.5)
        rho_out = channel.apply(rho_in)

        a = destroy(cutoff)
        n_hat = a.dag() * a
        assert abs(expect(n_hat, rho_in) - expect(n_hat, rho_out)) < 1e-7

    def test_fock_state_is_invariant(self, cutoff):
        """Fock states are eigenstates of n̂, so they are invariant under dephasing."""
        state = FockState(n=3, cutoff=cutoff)
        rho_in = state.density_matrix()
        channel = DephasingChannel(cutoff=cutoff, gamma=0.8)
        rho_out = channel.apply(rho_in)
        assert abs((rho_in - rho_out).norm()) < 1e-7

    def test_invalid_gamma_raises(self, cutoff):
        """Constructor must raise ValueError if gamma is negative."""
        with pytest.raises(ValueError):
            DephasingChannel(cutoff=cutoff, gamma=-0.5)

class TestDepolarizingChannel:
    def test_preserves_physicality(self, cutoff):
        """After depolarization, the state must remain physical (is_physical is True)."""
        state = CoherentState(alpha=1.0, cutoff=cutoff)
        channel = DepolarizingChannel(cutoff=cutoff, p=0.3)
        rho_out = channel.apply(state.density_matrix())
        assert is_physical(rho_out)

    def test_p_zero_is_identity(self, cutoff):
        """With p=0, the channel must return the original state unchanged."""
        state = FockState(n=2, cutoff=cutoff)
        rho_in = state.density_matrix()
        channel = DepolarizingChannel(cutoff=cutoff, p=0.0)
        rho_out = channel.apply(rho_in)
        assert abs((rho_in - rho_out).norm()) < 1e-10

    def test_p_one_is_maximally_mixed(self, cutoff):
        """With p=1, the channel must return the maximally mixed state I/d."""
        state = CoherentState(alpha=2.0, cutoff=cutoff)
        channel = DepolarizingChannel(cutoff=cutoff, p=1.0)
        rho_out = channel.apply(state.density_matrix())

        expected_identity = qeye(cutoff) / cutoff
        assert abs((rho_out - expected_identity).norm()) < 1e-10

    def test_invalid_p_raises(self, cutoff):
        """Constructor must raise ValueError if probability p is out of bounds."""
        with pytest.raises(ValueError):
            DepolarizingChannel(cutoff=cutoff, p=-0.1)
        with pytest.raises(ValueError):
            DepolarizingChannel(cutoff=cutoff, p=1.05)

class TestAmplificationChannel:
    def test_preserves_physicality(self, cutoff):
        """After amplification, the state must remain physical (is_physical is True)."""
        state = CoherentState(alpha=1.0, cutoff=cutoff)
        channel = AmplificationChannel(cutoff=cutoff, gamma=0.1)
        rho_out = channel.apply(state.density_matrix())
        assert is_physical(rho_out)

    def test_increases_photon_number(self, cutoff):
        """Amplification channel must increase the mean photon number.

        Amplification = photons are added to the system → ⟨n̂⟩_out > ⟨n̂⟩_in.
        """
        state = FockState(n=2, cutoff=cutoff)
        rho_in = state.density_matrix()

        channel = AmplificationChannel(cutoff=cutoff, gamma=0.2)
        rho_out = channel.apply(rho_in)

        a = destroy(cutoff)
        n_hat = a.dag() * a

        assert expect(n_hat, rho_out) > expect(n_hat, rho_in)

    def test_zero_amplification_preserves_state(self, cutoff):
        """With gamma=0, the channel acts as an identity operator (no changes)."""
        state = CoherentState(alpha=1.0, cutoff=cutoff)
        rho_in = state.density_matrix()

        channel = AmplificationChannel(cutoff=cutoff, gamma=0.0)
        rho_out = channel.apply(rho_in)

        assert abs((rho_in - rho_out).norm()) < 1e-7

    def test_invalid_gamma_raises(self, cutoff):
        """Constructor must raise ValueError if gamma is negative."""
        with pytest.raises(ValueError):
            AmplificationChannel(cutoff=cutoff, gamma=-0.2)

class TestGaussianNoiseChannel:
    def test_preserves_physicality(self, cutoff):
        """After Gaussian noise is added, the state must remain physical."""
        state = CoherentState(alpha=1.0, cutoff=cutoff)
        channel = GaussianNoiseChannel(cutoff=cutoff, gamma=0.1)
        rho_out = channel.apply(state.density_matrix())
        assert is_physical(rho_out)

    def test_increases_quadrature_variance(self, cutoff):
        """Gaussian noise channel must increase the variance of quadratures X and P."""
        state = FockState(n=0, cutoff=cutoff)
        rho_in = state.density_matrix()

        channel = GaussianNoiseChannel(cutoff=cutoff, gamma=0.1)
        rho_out = channel.apply(rho_in)

        a = destroy(cutoff)
        X = (a + a.dag()) / np.sqrt(2)

        var_x_in = expect(X ** 2, rho_in)
        var_x_out = expect(X ** 2, rho_out)

        assert var_x_out > var_x_in

    def test_adds_energy_to_system(self, cutoff):
        """Symmetric quadrature noise acts as heating, increasing the mean photon number."""
        state = CoherentState(alpha=1.5, cutoff=cutoff)
        rho_in = state.density_matrix()

        channel = GaussianNoiseChannel(cutoff=cutoff, gamma=0.15)
        rho_out = channel.apply(rho_in)

        a = destroy(cutoff)
        n_hat = a.dag() * a

        assert expect(n_hat, rho_out) > expect(n_hat, rho_in)

    def test_zero_noise_preserves_state(self, cutoff):
        """With gamma=0, the channel must act as an identity operator."""
        state = CoherentState(alpha=1.0, cutoff=cutoff)
        rho_in = state.density_matrix()

        channel = GaussianNoiseChannel(cutoff=cutoff, gamma=0.0)
        rho_out = channel.apply(rho_in)

        assert abs((rho_in - rho_out).norm()) < 1e-7

    def test_invalid_gamma_raises(self, cutoff):
        """Constructor must raise ValueError if gamma is negative."""
        with pytest.raises(ValueError):
            GaussianNoiseChannel(cutoff=cutoff, gamma=-0.1)

class TestThermalNoiseChannel:
    def test_preserves_physicality(self, cutoff):
        """After thermal noise is applied, the state must remain physical."""
        state = CoherentState(alpha=1.0, cutoff=cutoff)
        channel = ThermalNoiseChannel(cutoff=cutoff, gamma=0.1, n_th=0.5)
        rho_out = channel.apply(state.density_matrix())
        assert is_physical(rho_out)

    def test_vacuum_absorbs_thermal_photons(self, cutoff):
        """A vacuum state in a thermal bath must absorb energy (mean photon number increases)."""
        state = FockState(n=0, cutoff=cutoff)
        rho_in = state.density_matrix()

        channel = ThermalNoiseChannel(cutoff=cutoff, gamma=0.3, n_th=1.5)
        rho_out = channel.apply(rho_in)

        a = destroy(cutoff)
        n_hat = a.dag() * a

        assert expect(n_hat, rho_in) == 0.0
        assert expect(n_hat, rho_out) > 0.0
        assert expect(n_hat, rho_out) < 1.5

    def test_zero_gamma_preserves_state(self, cutoff):
        """With gamma=0, the channel must act as an identity operator regardless of n_th."""
        state = CoherentState(alpha=1.5, cutoff=cutoff)
        rho_in = state.density_matrix()

        channel = ThermalNoiseChannel(cutoff=cutoff, gamma=0.0, n_th=2.0)
        rho_out = channel.apply(rho_in)

        assert abs((rho_in - rho_out).norm()) < 1e-7

    def test_invalid_parameters_raise(self, cutoff):
        """Constructor must raise ValueError if gamma or n_th is negative."""
        with pytest.raises(ValueError):
            ThermalNoiseChannel(cutoff=cutoff, gamma=-0.1, n_th=0.5)
        with pytest.raises(ValueError):
            ThermalNoiseChannel(cutoff=cutoff, gamma=0.2, n_th=-0.5)
