import pytest
from qutip import destroy, expect

from src.physics import QuantumChannel, CoherentState, LossChannel, FockState, MixtureChannel, DephasingChannel
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
        """Po defazowaniu stan nadal musi być w pełni fizyczny."""
        state = CoherentState(alpha=1.5, cutoff=cutoff)
        channel = DephasingChannel(cutoff=cutoff, gamma=0.2)
        rho_out = channel.apply(state.density_matrix())
        assert is_physical(rho_out)

    def test_conserves_photon_number(self, cutoff):
        """Kluczowa cecha: defazowanie NIE zmienia średniej liczby fotonów!

        W przeciwieństwie do LossChannel, ewolucja pod operatorem n̂
        zmienia fazy, ale zachowuje rozkład prawdopodobieństwa obsadzeń.
        """
        state = CoherentState(alpha=2.0, cutoff=cutoff)
        rho_in = state.density_matrix()

        channel = DephasingChannel(cutoff=cutoff, gamma=0.5)
        rho_out = channel.apply(rho_in)

        a = destroy(cutoff)
        n_hat = a.dag() * a

        assert abs(expect(n_hat, rho_in) - expect(n_hat, rho_out)) < 1e-7

    def test_fock_state_is_invariant(self, cutoff):
        """Stan Focka |n⟩ jest stanem własnym n̂, więc defazowanie go nie zmienia."""
        state = FockState(n=3, cutoff=cutoff)
        rho_in = state.density_matrix()

        channel = DephasingChannel(cutoff=cutoff, gamma=0.8)
        rho_out = channel.apply(rho_in)

        # Różnica norm powinna być bliska zeru
        assert abs((rho_in - rho_out).norm()) < 1e-7

    def test_zero_dephasing_preserves_state(self, cutoff):
        """Dla gamma=0 kanał jest identycznością (brak zmian w stanie)."""
        state = CoherentState(alpha=1.5, cutoff=cutoff)
        rho_in = state.density_matrix()

        channel = DephasingChannel(cutoff=cutoff, gamma=0.0)
        rho_out = channel.apply(rho_in)

        assert abs((rho_in - rho_out).norm()) < 1e-7

    def test_invalid_gamma_raises(self, cutoff):
        """Konstruktor powinien odrzucić ujemną wartość parametru gamma."""
        with pytest.raises(ValueError):
            DephasingChannel(cutoff=cutoff, gamma=-0.5)
