import pytest
import numpy as np
from qutip import destroy, expect, basis
from src.physics import (
    FockState,
    CoherentState,
    CatState,
    ThermalState,
    VacuumState,
    BinomialState,
    GKPState,
)
from src.physics.validation import is_physical

pure_state_factories = [
    lambda c: FockState(n=3, cutoff=c),
    lambda c: CoherentState(alpha=1.5, cutoff=c),
    lambda c: CatState(alpha=2.0, cutoff=c),
    lambda c: VacuumState(cutoff=c),
    lambda c: BinomialState(N=4, p=0.5, cutoff=c),
    lambda c: GKPState(cutoff=c, delta=0.3)
]

@pytest.mark.parametrize("state_factory", pure_state_factories)
def test_common_pure_state_properties(state_factory, cutoff):
    """Automatyczny test dla wszystkich stanów czystych z tabeli wymagań."""
    state = state_factory(cutoff)
    rho = state.density_matrix()

    # Normalizacja ketu | ⟨ψ|ψ⟩ = 1
    assert abs(state.ket().norm() - 1) < 1e-10

    # Ślad macierzy gęstości | Tr(ρ) = 1
    assert abs(rho.tr() - 1) < 1e-10

    # Hermitowskość | ρ = ρ†
    assert rho.isherm is True

    # Dodatnia półokreśloność | wartości własne ≥ 0
    assert np.all(rho.eigenenergies() >= -1e-10)

    # Kompleksowa walidacja funkcją systemową
    assert is_physical(rho) is True


def test_common_mixed_state_properties(cutoff):
    """Weryfikacja warunków fizycznych dla stanu mieszanego (ThermalState).

    Stan termiczny nie ma reprezentacji wektorowej (ketu),
    więc testujemy tylko właściwości macierzy gęstości.
    """
    state = ThermalState(n_th=1.5, cutoff=cutoff)
    rho = state.density_matrix()

    assert abs(rho.tr() - 1) < 1e-10
    assert rho.isherm is True
    assert np.all(rho.eigenenergies() >= -1e-10)
    assert is_physical(rho) is True


class TestFockState:
    def test_photon_number(self, cutoff):
        """Stan Focka |n⟩ powinien mieć dokładnie n fotonów.

        Jak to działa:
        - a = destroy() to operator anihilacji (patrz sekcja 1a)
        - n_hat = a.dag() * a to operator liczby fotonów (n̂ = a†a)
        - expect(n_hat, rho) oblicza wartość oczekiwaną ⟨n̂⟩ — średnią liczbę fotonów
        - Dla stanu Focka |3⟩ oczekujemy ⟨n̂⟩ = 3.0 (dokładnie 3 fotony)
        """
        state = FockState(n=3, cutoff=cutoff)
        rho = state.density_matrix()
        a = destroy(cutoff)
        n_hat = a.dag() * a
        assert abs(expect(n_hat, rho) - 3.0) < 1e-10


class TestCoherentState:
    def test_mean_photon_number(self, cutoff):
        """Stan koherentny |α⟩ powinien mieć średnio |α|² fotonów.

        Jak to działa:
        - Dla stanu koherentnego z α=2.0 oczekujemy ⟨n̂⟩ = |2.0|² = 4.0
        - Tolerancja 0.01 (a nie 1e-10) bo cutoff obcina przestrzeń Hilberta
        """
        alpha = 2.0
        state = CoherentState(alpha=alpha, cutoff=cutoff)
        rho = state.density_matrix()
        a = destroy(cutoff)
        n_hat = a.dag() * a
        assert abs(expect(n_hat, rho) - abs(alpha) ** 2) < 0.01


class TestCatState:
    def test_symmetry(self, cutoff):
        """Parzysty stan kota powinien zawierać tylko parzyste stany Focka.

        Jak to działa:
        - Kot |α⟩ + |-α⟩ to superpozycja symetryczna (parzysty kot)
        - Po rozłożeniu na stany Focka zawiera tylko |0⟩, |2⟩, |4⟩...
        - Nieparzyste składowe (|1⟩, |3⟩, |5⟩) powinny mieć zerowy wkład
        - basis(cutoff, n).dag() * psi oblicza „nakładanie" stanu kota na |n⟩
        """
        state = CatState(alpha=2.0, cutoff=cutoff)
        psi = state.ket()

        for n in [1, 3, 5]:
            overlap = abs(basis(cutoff, n).dag() * psi)
            assert abs(overlap) < 1e-10


class TestThermalState:
    def test_mean_photon_number(self, cutoff):
        """Stan termiczny powinien mieć średnio n_th fotonów.

        Stan termiczny opisuje światło w równowadze cieplnej.
        Parametr n_th to średnia liczba fotonów (rozkład Bosego-Einsteina).
        """
        n_th = 2.0
        state = ThermalState(n_th=n_th, cutoff=cutoff)
        rho = state.density_matrix()
        a = destroy(cutoff)
        n_hat = a.dag() * a
        assert abs(expect(n_hat, rho) - n_th) < 0.1

    def test_ket_raises(self, cutoff):
        """Stan termiczny jest stanem mieszanym — nie ma ketu.

        Próba wywołania ket() powinna zgłosić błąd NotImplementedError,
        bo stanu mieszanego nie da się opisać pojedynczym wektorem stanu.
        """
        state = ThermalState(n_th=1.0, cutoff=cutoff)
        with pytest.raises(NotImplementedError):
            state.ket()


class TestVacuumState:
    def test_zero_photons(self, cutoff):
        """Próżnia |0⟩ powinna mieć 0 fotonów."""
        state = VacuumState(cutoff=cutoff)
        rho = state.density_matrix()
        a = destroy(cutoff)
        n_hat = a.dag() * a
        assert abs(expect(n_hat, rho)) < 1e-10


class TestBinomialState:
    def test_normalization(self, cutoff):
        """Stan dwumianowy powinien być znormalizowany (⟨ψ|ψ⟩ = 1)."""
        state = BinomialState(N=4, p=0.5, cutoff=cutoff)
        assert abs(state.ket().norm() - 1) < 1e-10

    def test_invalid_p(self, cutoff):
        """Prawdopodobieństwo p musi być w zakresie [0, 1].

        Parametr p=1.5 jest niefizyczny — konstruktor powinien to odrzucić.
        """
        with pytest.raises(ValueError):
            BinomialState(N=4, p=1.5, cutoff=cutoff)


class TestGKPState:
    def test_normalization(self, cutoff):
        """Stan GKP powinien być znormalizowany (⟨ψ|ψ⟩ = 1)."""
        state = GKPState(cutoff=cutoff, delta=0.3)
        assert abs(state.ket().norm() - 1) < 1e-10
