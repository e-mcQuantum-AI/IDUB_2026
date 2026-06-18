import pytest
from qutip import basis

from src.physics import FockState, ThermalState
from src.physics.validation import is_physical


class TestIsPhysical:
    def test_valid_pure_state(self, cutoff):
        """Macierz gęstości stanu czystego powinna być fizyczna."""
        rho = FockState(n=0, cutoff=cutoff).density_matrix()
        assert is_physical(rho)

    def test_valid_mixed_state(self, cutoff):
        """Macierz gęstości stanu termicznego (mieszanego) powinna być fizyczna."""
        rho = ThermalState(n_th=1.0, cutoff=cutoff).density_matrix()
        assert is_physical(rho)

    def test_non_hermitian(self, cutoff):
        """Macierz niehermitowska NIE powinna być fizyczna.

        Dodajemy urojoną składową, która łamie symetrię ρ = ρ†.
        """
        rho = FockState(n=0, cutoff=cutoff).density_matrix()
        rho_bad = rho + 0.01j * basis(cutoff, 0) * basis(cutoff, 1).dag()
        assert not is_physical(rho_bad)

    def test_wrong_trace(self, cutoff):
        """Macierz ze śladem ≠ 1 NIE powinna być fizyczna.

        Mnożymy macierz przez 2 → ślad = 2 (prawdopodobieństwa sumują się do 200%).
        """
        rho = 2 * FockState(n=0, cutoff=cutoff).density_matrix()
        assert not is_physical(rho)

    def test_negative_eigenvalue(self, cutoff):
        """Macierz z ujemną wartością własną NIE powinna być fizyczna.

        Ręcznie tworzymy macierz diagonalną z wartościami [1.5, -0.5, 0, ...].
        Ma ślad = 1 i jest hermitowska, ale wartość -0.5 oznaczałaby
        ujemne prawdopodobieństwo — to niefizyczne.
        """
        import numpy as np
        from qutip import Qobj
        mat = np.diag([1.5, -0.5] + [0.0] * (cutoff - 2))
        rho_bad = Qobj(mat)
        assert not is_physical(rho_bad)
