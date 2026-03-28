# Raport o stanie projektu IDUB_2026

**Projekt**: Głębokie uczenie do klasyfikacji i rekonstrukcji kwantowych stanów optycznych
**Repozytorium**: https://github.com/e-mcQuantum-AI/IDUB_2026
**Data raportu**: 28.03.2026

---

## 1. Opis projektu

Projekt ma na celu stworzenie narzędzia do generowania syntetycznych danych obrazowych (funkcji Wignera) kwantowych stanów optycznych, które posłużą jako zbiór treningowy do klasyfikacji i rekonstrukcji stanów za pomocą głębokiego uczenia.

Obecny kod obejmuje:
- Implementację 7 stanów kwantowych (Fock, koherentny, kot Schrödingera, GKP, dwumianowy, termiczny, próżnia)
- Kanały szumu kwantowego (straty fotonowe, mieszanina stanów)
- Pomiar funkcją Wignera z potokiem pomiarowym (pipeline)
- Skrypty do generowania i dzielenia obrazów zbioru danych

**Rozmiar kodu**: ~330 linii Pythona w 18 plikach.

---

## 1a. Wprowadzenie do pojęć — dla początkujących

Ten rozdział wyjaśnia kluczowe pojęcia z fizyki kwantowej i programowania, które pojawiają się w dalszej części raportu. Jeśli znasz te zagadnienia, możesz przejść do sekcji 2.

### Pojęcia z fizyki kwantowej

**Stan kwantowy** — matematyczny opis układu kwantowego. W tym projekcie opisujemy stany światła (fotony w jednym modzie pola elektromagnetycznego).

**Ket |ψ⟩** (czytaj: „ket psi") — wektor opisujący stan kwantowy. Zapisujemy go w notacji Diraca: `|ψ⟩`. Np. `|3⟩` oznacza stan z dokładnie 3 fotonami. W kodzie ket to wektor kolumnowy (obiekt `Qobj` z biblioteki QuTiP).

**Bra ⟨ψ|** — wektor sprzężony do ketu (`|ψ⟩` transponowany i zespolenie sprzężony). Potrzebny do obliczeń typu iloczyn skalarny.

**Iloczyn skalarny ⟨ψ|ψ⟩** — miara „nakładania się" dwóch stanów. Jeśli `⟨ψ|ψ⟩ = 1`, stan jest **znormalizowany** (ma sens fizyczny). W kodzie: `state.ket().norm()`.

**Stan czysty vs. stan mieszany**:
- **Stan czysty** — układ jest w jednym, określonym stanie kwantowym `|ψ⟩`. Przykład: dokładnie 3 fotony.
- **Stan mieszany** — układ jest w jednym z kilku stanów z pewnym prawdopodobieństwem (mieszanina statystyczna). Przykład: stan termiczny — nie wiemy ile jest fotonów, znamy tylko średnią. Stanu mieszanego **nie da się** opisać pojedynczym ketem.

**Macierz gęstości ρ** (rho) — uniwersalny sposób opisu stanu kwantowego, działa zarówno dla stanów czystych jak i mieszanych:
- Dla stanu czystego: `ρ = |ψ⟩⟨ψ|` (iloczyn zewnętrzny ketu i bra)
- Dla stanu mieszanego: `ρ = Σ p_i |ψ_i⟩⟨ψ_i|` (suma ważona prawdopodobieństwami)
- W kodzie: `state.density_matrix()` — zwraca macierz kwadratową.

**Fizyczność macierzy gęstości** — poprawna macierz gęstości musi spełniać 3 warunki:
1. **Hermitowskość** (`ρ = ρ†`) — macierz jest równa swojej transpozycji zespolenie sprzężonej. W kodzie: `rho.isherm`.
2. **Ślad = 1** (`Tr(ρ) = 1`) — suma elementów na przekątnej wynosi 1 (prawdopodobieństwa sumują się do 1). W kodzie: `rho.tr()`.
3. **Dodatnia półokreśloność** — wszystkie wartości własne macierzy są ≥ 0 (prawdopodobieństwa nie mogą być ujemne). W kodzie: `rho.eigenenergies()`.

**Przestrzeń Hilberta** — przestrzeń matematyczna, w której „żyją" stany kwantowe. Wymiar tej przestrzeni jest teoretycznie nieskończony (foton może mieć dowolną liczbę), ale w komputerze musimy ją **obciąć** — parametr `cutoff` określa, ile stanów bazowych (0, 1, 2, ..., cutoff-1 fotonów) uwzględniamy. Np. `cutoff=32` oznacza, że rozważamy stany od 0 do 31 fotonów.

**Operator kreacji a† i anihilacji a** — operatory matematyczne na przestrzeni Hilberta:
- `a|n⟩ = √n |n-1⟩` — anihilacja „zabiera" foton
- `a†|n⟩ = √(n+1) |n+1⟩` — kreacja „dodaje" foton
- `n̂ = a†a` — **operator liczby fotonów**: `n̂|n⟩ = n|n⟩`
- W kodzie: `destroy()` = a, `create()` = a†.

**Funkcja Wignera** — sposób wizualizacji stanu kwantowego jako „mapy cieplnej" w przestrzeni fazowej (oś x = pozycja q, oś y = pęd p). Każdy typ stanu ma charakterystyczny wzór na wykresie Wignera — to właśnie te wzory projekt chce klasyfikować za pomocą ML. Funkcja Wignera może przyjmować wartości ujemne — to cecha typowo kwantowa, niemożliwa w klasycznej fizyce.

**Operator przesunięcia D(α)** — przesuwa stan w przestrzeni fazowej o wektor α (liczba zespolona: część rzeczywista = przesunięcie w q, część urojona = przesunięcie w p). W kodzie: `displace(cutoff, alpha)`.

### Stany kwantowe zaimplementowane w projekcie

| Stan | Symbol | Opis intuicyjny | Przykład z życia |
|------|--------|-----------------|-----------------|
| **Próżnia** | \|0⟩ | Brak fotonów — ale nie brak energii! Próżnia kwantowa ma niezerowe fluktuacje. | „Ciemność" w doskonale odizolowanym pudełku |
| **Fock** | \|n⟩ | Dokładnie n fotonów. Stan o określonej liczbie cząstek. | Pojedynczy foton z emitera kwantowego |
| **Koherentny** | \|α⟩ | Stan najbliższy klasycznemu światłu. Parametr α określa amplitudę. Średnia liczba fotonów = \|α\|². | Światło lasera |
| **Kot Schrödingera** | \|α⟩ + \|-α⟩ | Superpozycja dwóch stanów koherentnych o przeciwnych amplitudach — „jednocześnie tu i tam" w przestrzeni fazowej. | Eksperyment myślowy Schrödingera przeniesiony do optyki |
| **Termiczny** | ρ_th | Stan mieszany (nie da się opisać ketem). Opisuje światło w równowadze termicznej — rozkład Bosego-Einsteina. Parametr n_th = średnia liczba fotonów. | Promieniowanie cieplne żarówki |
| **Dwumianowy** | \|ψ_bin⟩ | Superpozycja stanów Focka ze współczynnikami dwumianowymi. Kod korekcji błędów kwantowych. | Badania nad pamięciami kwantowymi |
| **GKP** | \|GKP⟩ | Kod korekcji błędów: kubit zakodowany w oscylatorze harmonicznym. Regularna siatka pików w przestrzeni fazowej. | Komputery kwantowe oparte na zmiennych ciągłych |

### Kanały szumu kwantowego

Kanał szumu to operacja, która symuluje niedoskonałości fizyczne — zmienia stan kwantowy w sposób nieodwracalny (wprowadza szum):

| Kanał | Co robi | Analogia |
|-------|---------|----------|
| **Straty fotonowe** (loss) | Fotony „uciekają" z układu z prawdopodobieństwem γ. Zmniejsza średnią liczbę fotonów. | Światło przechodzące przez brudne szkło |
| **Mieszanina** (mixture) | Miesza dwa stany: `ρ' = p·ρ + (1-p)·ρ_other`. | Detektor, który czasem widzi inny stan |

### Pojęcia programistyczne

**`__init__.py`** — specjalny plik w Pythonie, który mówi interpreterowi: „ten katalog jest pakietem (modułem) Pythona". Bez niego nie można pisać `from src.physics import FockState`. Plik może być pusty lub zawierać eksporty.

**Klasa abstrakcyjna (ABC)** — klasa, która definiuje „umowę" (interfejs): mówi jakie metody muszą mieć klasy potomne, ale sama ich nie implementuje. Np. `QuantumState` wymaga metody `ket()` — każdy konkretny stan (Fock, koherentny...) musi ją zaimplementować po swojemu.

**Dziedziczenie** — mechanizm, w którym klasa potomna przejmuje cechy klasy nadrzędnej. `class LossChannel(QuantumChannel)` oznacza: „LossChannel jest rodzajem QuantumChannel i musi spełniać jego kontrakt". Jeśli brakuje dziedziczenia, Python nie wie, że LossChannel jest kanałem kwantowym.

**Adnotacje typów (type hints)** — podpowiedzi dla programisty i narzędzi, jakiego typu są argumenty i wyniki funkcji: `def ket(self) -> Qobj:` mówi „ta metoda zwraca obiekt typu Qobj". Python ich nie wymusza, ale pomagają w wykrywaniu błędów.

**Docstring** — komentarz dokumentacyjny wewnątrz funkcji/klasy (w potrójnych cudzysłowach `"""`). Opisuje co robi, jakie argumenty przyjmuje, co zwraca. Narzędzia mogą go automatycznie wyciągnąć do dokumentacji.

**`requirements.txt`** — plik z listą bibliotek, od których zależy projekt (np. `qutip>=5.0`). Pozwala każdemu zainstalować wszystko jednym poleceniem: `pip install -r requirements.txt`.

**pytest / fixture** — `pytest` to framework do pisania testów automatycznych. `fixture` to przygotowane dane/obiekty, które testy mogą współdzielić (np. `cutoff=32` używany przez wiele testów).

**Pipeline** — wzorzec programistyczny „potok": dane przechodzą kolejno przez etapy przetwarzania. Tu: `stan → (opcjonalnie) szum → pomiar Wignera → obraz`.

**PEP8** — oficjalny przewodnik stylu kodu Pythona (nazwy zmiennych, wcięcia, długość linii). Spójny styl ułatwia czytanie kodu przez zespół.

**CI/CD (Continuous Integration)** — automatyczne uruchamianie testów przy każdym `git push`. Jeśli testy nie przejdą, zespół od razu wie, że coś się zepsuło.

---

## 2. Ocena obecnego stanu

### 2.1 Architektura — ocena: DOBRA

Projekt ma przejrzystą strukturę obiektową z użyciem klas abstrakcyjnych (ABC):

```
src/physics/
├── state/          — implementacje stanów kwantowych
├── noise/          — kanały szumowe
├── measurement/    — pomiar funkcją Wignera + pipeline
├── divider.py      — narzędzie do dzielenia obrazów
├── generate.py     — skrypt generujący zbiór danych
├── hilbert.py      — operatory przestrzeni Hilberta
└── validation.py   — walidacja stanów fizycznych
```

Dobór bibliotek jest trafny: **QuTiP**, NumPy, SciPy, Matplotlib, PIL.

### 2.2 Poprawność fizyczna — ocena: DOBRA z zastrzeżeniami

| Stan | Implementacja | Poprawność |
|------|---------------|------------|
| Próżnia (`vacuum.py`) | `basis(N, 0)` | ✓ Poprawna |
| Fock (`fock.py`) | `basis(N, n)` | ✓ Poprawna |
| Koherentny (`coherent.py`) | QuTiP `coherent()` | ✓ Poprawna |
| Kot (`cat.py`) | superpozycja stanów koherentnych | ✓ Poprawna |
| Termiczny (`thermal.py`) | `thermal_dm()`, stan mieszany | ✓ Poprawna |
| GKP (`gkp.py`) | obwiednia gaussowska + przesunięcia | ⚠ Wymaga weryfikacji wzoru |
| Dwumianowy (`binomal.py`) | współczynniki dwumianowe | ⚠ Wymaga weryfikacji definicji |

Walidacja macierzy gęstości (hermitowskość, ślad = 1, dodatnia półokreśloność) jest poprawna.

### 2.3 Jakość kodu — ocena: PRZECIĘTNA

| Aspekt | Ocena | Uwagi |
|--------|-------|-------|
| Struktura katalogów | Dobra | Logiczny podział na moduły |
| Czytelność | Dobra | Krótkie, jasne funkcje |
| Styl (PEP8) | Dobra | Generalnie zgodny |
| Dokumentacja | Słaba | Brak docstringów we wszystkich modułach |
| Typowanie | Brak | Zero adnotacji typów |
| Obsługa błędów | Brak | Brak walidacji danych wejściowych |
| Testy | Brak | Zero testów |
| Konfiguracja | Słaba | Ścieżki i parametry wpisane na sztywno |

---

## 3. Wykryte błędy i problemy

### 3.1 Błędy krytyczne

**B1. Brak plików `__init__.py`**
Żaden pakiet nie ma pliku `__init__.py`. Import modułów przez `from src.physics import ...` jest niemożliwy.
- Dotyczy: `src/`, `src/physics/`, `src/physics/state/`, `src/physics/noise/`, `src/physics/measurement/`

**B2. Błąd matematyczny w `generate.py` (linia 44)**
```python
state_binomial = (fock(N, n) + p * fock(N, n+1)).unit()
```
To NIE jest stan dwumianowy — to superpozycja dwóch stanów Focka |n⟩ + p|n+1⟩. Prawdziwy stan dwumianowy powinien używać współczynników dwumianowych (tak jak w `binomal.py`). Kod w `generate.py` jest niespójny z implementacją w `state/binomal.py`.

**B3. Literówka w nazwie pliku: `binomal.py`**
Plik `state/binomal.py` powinien nazywać się `binomial.py`.

**B4. Kanały szumu nie dziedziczą po klasie bazowej**
`LossChannel` i `MixtureChannel` nie dziedziczą po `QuantumChannel` (zdefiniowanej w `noise/base.py`). Oznacza to, że Python nie wie, iż te klasy są kanałami kwantowymi — nie można ich używać zamiennie w kodzie, który oczekuje obiektu typu `QuantumChannel` (np. w `MeasurementPipeline`).

### 3.2 Problemy istotne

**P1. Ścieżki wpisane na sztywno**
- `divider.py`: `'quantum_dataset/clean/'`
- `generate.py`: `'quantum_dataset/'`
- Brak przenośności między maszynami.

**P2. Brak `requirements.txt`**
Zależności projektu (QuTiP, NumPy, SciPy, Matplotlib, Pillow, OpenCV) nie są nigdzie udokumentowane.

**P3. Brak walidacji danych wejściowych**
- `BinomialState`: brak sprawdzenia, czy `p ∈ [0, 1]` i `N ≥ 0`
- `MixtureChannel`: brak sprawdzenia, czy `p ∈ [0, 1]`
- `LossChannel`: brak sprawdzenia, czy `gamma > 0`

**P4. Wzór GKP (`gkp.py:25`) nieudokumentowany**
```python
weight = np.exp(- (2 * s * np.sqrt(np.pi))**2 / (2 * self.delta**2))
```
Stan GKP (Gottesman-Kitaev-Preskill) to kwantowy kod korekcji błędów kodujący kubit w oscylatorze harmonicznym. Idealny stan GKP to nieskończony grzebień delt Diraca w przestrzeni pozycji, rozmieszczonych co 2√π. W praktyce stosuje się przybliżenie z obwiednią gaussowską o szerokości δ. Brak referencji do publikacji źródłowej w kodzie utrudnia weryfikację poprawności implementacji.

Referencja: D. Gottesman, A. Kitaev, J. Preskill, *„Encoding a qubit in an oscillator"*, Phys. Rev. A **64**, 012310 (2001), https://doi.org/10.1103/PhysRevA.64.012310

**P5. Mieszanie języków**
Komentarze w `generate.py` po polsku, reszta kodu po angielsku.

### 3.3 Problemy mniejsze

- **P6.** Stała magiczna `-1e-10` w `validation.py` — powinna być konfigurowalna
- **P7.** Parametr `tlist=[0, 1]` w `loss.py` — brak wyjaśnienia jednostek czasu
- **P8.** Brak logowania — tylko `print()` w `generate.py`

---

## 4. Plan kolejnych działań

### Faza 1 — Naprawa błędów krytycznych (priorytet: WYSOKI)

| # | Zadanie | Plik(i) | Szacowany czas |
|---|---------|---------|----------------|
| 1.1 | Dodać pliki `__init__.py` do wszystkich pakietów | 5 plików | 30 min |
| 1.2 | Poprawić nazwę pliku `binomal.py` → `binomial.py` | `state/binomal.py` | 15 min |
| 1.3 | Dodać dziedziczenie `LossChannel(QuantumChannel)` i `MixtureChannel(QuantumChannel)` | `noise/loss.py`, `noise/mixture.py` | 30 min |
| 1.4 | Naprawić generowanie stanu dwumianowego w `generate.py` — użyć klasy `BinomialState` | `generate.py` | 1 godz. |
| 1.5 | Zweryfikować wzory matematyczne: stan dwumianowy i stan GKP — porównać z literaturą | `state/binomal.py`, `state/gkp.py` | 2 godz. |

### Faza 2 — Infrastruktura projektu (priorytet: WYSOKI)

| # | Zadanie | Szacowany czas |
|---|---------|----------------|
| 2.1 | Utworzyć `requirements.txt` z wersjami zależności | 30 min |
| 2.2 | Sparametryzować ścieżki i stałe — wprowadzić plik konfiguracyjny lub argumenty CLI (`argparse`) | 2 godz. |
| 2.3 | Ujednolicić język komentarzy (angielski) | 30 min |
| 2.4 | Rozbudować `README.md` — opis projektu, instalacja, przykłady użycia | 2 godz. |

### Faza 3 — Jakość kodu (priorytet: ŚREDNI)

| # | Zadanie | Szacowany czas |
|---|---------|----------------|
| 3.1 | Dodać adnotacje typów do wszystkich funkcji i metod | 3 godz. |
| 3.2 | Dodać docstringi do wszystkich klas i metod publicznych | 4 godz. |
| 3.3 | Dodać walidację danych wejściowych (parametry stanów, kanałów) | 2 godz. |
| 3.4 | Dodać obsługę błędów w `divider.py` i `generate.py` | 1 godz. |
| 3.5 | Dodać referencje do literatury w docstringach stanów GKP i dwumianowego | 1 godz. |

### Faza 4 — Testy (priorytet: ŚREDNI)

| # | Zadanie | Szacowany czas |
|---|---------|----------------|
| 4.1 | Skonfigurować `pytest` i utworzyć katalog `tests/` | 1 godz. |
| 4.2 | Testy jednostkowe stanów: poprawność wektora stanu, macierzy gęstości, normalizacja | 4 godz. |
| 4.3 | Testy kanałów szumu: zachowanie fizyczności stanu po zastosowaniu kanału | 2 godz. |
| 4.4 | Testy walidacji: sprawdzenie `is_physical()` na stanach poprawnych i niepoprawnych | 1 godz. |
| 4.5 | Testy integracyjne pipeline: stan → szum → pomiar → obraz | 2 godz. |

### Faza 5 — Rozwój funkcjonalności (priorytet: NISKI)

| # | Zadanie | Szacowany czas |
|---|---------|----------------|
| 5.1 | Dodać brakujące kanały szumu: defazowanie (dephasing), wzmocnienie (amplification) | 3 godz. |
| 5.2 | Dodać możliwość generowania stanów mieszanych (nie tylko czystych) | 2 godz. |
| 5.3 | Skonfigurować CI/CD (GitHub Actions) z automatycznym uruchamianiem testów | 2 godz. |
| 5.4 | Zintegrować moduł ML — klasyfikator stanów kwantowych na podstawie obrazów Wignera | Otwarte |
| 5.5 | Zoptymalizować generowanie stanów GKP (cache operatorów przesunięcia) | 2 godz. |

---

## 5. Szczegółowe plany działań

### Faza 1 — Naprawa błędów krytycznych

#### 1.1 Dodanie plików `__init__.py`

**Problem**: Brak plików `__init__.py` uniemożliwia import pakietów Pythona.

**Kroki**:
1. Utworzyć puste pliki `__init__.py` w katalogach:
   - `src/__init__.py`
   - `src/physics/__init__.py`
   - `src/physics/state/__init__.py`
   - `src/physics/noise/__init__.py`
   - `src/physics/measurement/__init__.py`
2. W `src/physics/__init__.py` dodać eksporty głównych klas:
   ```python
   from .state.fock import FockState
   from .state.coherent import CoherentState
   from .state.cat import CatState
   from .state.vacuum import VacuumState
   from .state.thermal import ThermalState
   from .state.gkp import GKPState
   from .state.binomial import BinomialState
   from .noise.loss import LossChannel
   from .noise.mixture import MixtureChannel
   from .measurement.wigner import WignerMeasurement
   from .measurement.pipeline import MeasurementPipeline
   ```
3. Poprawić niespójne importy — w `coherent.py` i `vacuum.py` jest `from src.physics.state.base import QuantumState`, a w pozostałych plikach `from .base import QuantumState`. Ujednolicić do importów względnych (`from .base`).
4. Sprawdzić, czy `from src.physics import FockState` działa poprawnie.

**Kryterium zakończenia**: Polecenie `python -c "from src.physics import FockState"` wykonuje się bez błędów.

---

#### 1.2 Zmiana nazwy `binomal.py` → `binomial.py`

**Problem**: Literówka w nazwie pliku.

**Kroki**:
1. Zmienić nazwę: `git mv src/physics/state/binomal.py src/physics/state/binomial.py`
2. Zaktualizować wszystkie importy odnoszące się do `binomal` → `binomial` (w tym przyszły `__init__.py`).
3. Sprawdzić, czy nie ma innych odwołań do starej nazwy (`grep -r "binomal"`).

**Kryterium zakończenia**: Brak pliku `binomal.py`, import `from src.physics.state.binomial import BinomialState` działa.

---

#### 1.3 Dodanie dziedziczenia w kanałach szumu

**Problem**: `LossChannel` i `MixtureChannel` nie dziedziczą po `QuantumChannel`, łamiąc kontrakt interfejsu.

**Kroki**:
1. W `noise/loss.py` zmienić deklarację klasy:
   ```python
   # PRZED:
   class LossChannel:
   # PO:
   from .base import QuantumChannel
   class LossChannel(QuantumChannel):
   ```
2. W `noise/mixture.py` analogicznie:
   ```python
   from .base import QuantumChannel
   class MixtureChannel(QuantumChannel):
   ```
3. Sprawdzić, czy metoda `apply()` w obu klasach zachowuje sygnaturę z klasy bazowej `apply(self, rho: Qobj) -> Qobj`.
4. Przetestować ręcznie: utworzyć stan, zastosować kanał, sprawdzić wynik.

**Kryterium zakończenia**: `isinstance(LossChannel(32, 0.1), QuantumChannel)` zwraca `True`.

---

#### 1.4 Naprawa generowania stanu dwumianowego w `generate.py`

**Problem**: Linia 44 w `generate.py` tworzy superpozycję dwóch stanów Focka `|n⟩ + p|n+1⟩` zamiast prawdziwego stanu dwumianowego.

**Kroki**:
1. Zrefaktoryzować `generate.py`, aby używał klas z pakietu `state/` zamiast ręcznego konstruowania stanów:
   ```python
   # PRZED (linia 44):
   state_binomial = (fock(N, n) + p * fock(N, n+1)).unit()

   # PO:
   from src.physics.state.binomial import BinomialState
   state_binomial = BinomialState(N=n, p=p, cutoff=N).ket()
   ```
2. Rozważyć refaktoryzację całego `generate.py`, aby korzystał z istniejących klas `FockState`, `CoherentState`, `CatState` zamiast bezpośrednio z QuTiP — zapewni to spójność.
3. Zweryfikować wizualnie wygenerowane obrazy Wignera stanu dwumianowego — porównać z literaturą (np. M. Bergmann, P. van Loock, „Quantum error correction against photon loss using NOON states", PRA 2016).

**Kryterium zakończenia**: `generate.py` używa klasy `BinomialState`, generowane obrazy są fizycznie poprawne.

---

#### 1.5 Weryfikacja wzorów matematycznych

**Problem**: Wzory w `BinomialState` i `GKPState` nie są udokumentowane i wymagają porównania z literaturą.

##### Stan dwumianowy (`binomial.py`)

Obecna implementacja:
```python
coeff = sqrt(C(N, n)) * p^(n/2) * (1-p)^((N-n)/2)
|ψ⟩ = Σ coeff_n |n⟩
```

**Kroki weryfikacji**:
1. Sprawdzić definicję w: M. H. Michael et al., „New class of quantum error-correcting codes for a bosonic mode", PRX 6, 031006 (2016).
2. Standardowa definicja stanu dwumianowego:
   ```
   |ψ⟩ = Σ_{n=0}^{N} sqrt(C(N,n)) * p^(n/2) * (1-p)^((N-n)/2) |n⟩
   ```
   To zgadza się z implementacją — ale trzeba potwierdzić, że projekt odnosi się do tej konkretnej definicji.
3. Sprawdzić normalizację: `⟨ψ|ψ⟩ = Σ C(N,n) * p^n * (1-p)^(N-n) = (p + 1-p)^N = 1` — teoretycznie poprawna, `.unit()` jest nadmiarowe ale nie szkodzi.
4. Upewnić się, że `cutoff > N` (inaczej pętla obcina stany).

##### Stan GKP (`gkp.py`)

**Kontekst fizyczny**: Stan GKP (Gottesman-Kitaev-Preskill) to kwantowy kod korekcji błędów dla zmiennych ciągłych, zaproponowany w 2001 roku. Koduje kubit logiczny w oscylatorze harmonicznym (np. modzie pola elektromagnetycznego), chroniąc go przed małymi błędami przesunięcia w pozycji (q) i pędzie (p).

Idealny stan GKP to nieskończony grzebień delt Diraca w przestrzeni pozycji:
```
|0_GKP⟩ ∝ Σ_s  δ(q - 2s√π)       — piki w pozycjach parzystych
|1_GKP⟩ ∝ Σ_s  δ(q - (2s+1)√π)   — piki w pozycjach nieparzystych
```

Idealny stan wymaga nieskończonej energii, dlatego w implementacji stosuje się **przybliżony stan GKP** z obwiednią gaussowską o szerokości δ:

Obecna implementacja:
```python
weight = exp(-(2s√π)² / (2δ²))        # obwiednia gaussowska
|GKP⟩ ∝ Σ_s  weight · D(2s√π/√2) |0⟩  # suma ważonych przesuniętych stanów próżni
```

Parametr `δ` kontroluje kompromis: mniejszy δ → bliżej ideału, ale większa wymagana energia i wymiarowość przestrzeni Hilberta. Na wykresie funkcji Wignera stan GKP wyróżnia się charakterystyczną **regularną siatką pików**.

**Referencja źródłowa**: D. Gottesman, A. Kitaev, J. Preskill, *„Encoding a qubit in an oscillator"*, Phys. Rev. A **64**, 012310 (2001), https://doi.org/10.1103/PhysRevA.64.012310

**Kroki weryfikacji**:
1. Porównać implementację z równaniem (18) w powyższej publikacji — przybliżony stan GKP z gaussowskim wygładzeniem.
2. Sprawdzić, czy argument operatora przesunięcia `q_shift / sqrt(2)` poprawnie przelicza pozycję na amplitudę zespoloną (konwencja QuTiP: `D(α)` przesuwa o `Re(α)` w q i `Im(α)` w p).
3. Sprawdzić, czy obwiednia gaussowska `exp(-q²/(2δ²))` ma poprawną parametryzację — w obecnym kodzie `q = 2s√π`, więc `weight = exp(-(2s√π)²/(2δ²))`, co odpowiada gaussowskiemu wygładzeniu z szerokością `δ` w przestrzeni pozycji.
4. Wygenerować funkcję Wignera stanu GKP i porównać wizualnie z rysunkami z publikacji (oczekiwany wzór: regularna siatka pików w przestrzeni fazowej).

**Kryterium zakończenia**: Obie implementacje mają referencje do publikacji w docstringach. Poprawność potwierdzona analitycznie lub numerycznie.

---

### Faza 2 — Infrastruktura projektu

#### 2.1 Utworzenie `requirements.txt`

**Kroki**:
1. Zidentyfikować wszystkie zależności na podstawie importów w kodzie:
   - `qutip` — obliczenia kwantowe
   - `numpy` — obliczenia numeryczne
   - `scipy` — funkcje specjalne (`comb`)
   - `matplotlib` — generowanie obrazów
   - `Pillow` — manipulacja obrazami (`PIL`)
   - `opencv-python` — przetwarzanie obrazów (tylko `img_cutter.ipynb`)
2. Utworzyć plik `requirements.txt` w katalogu głównym:
   ```
   qutip>=5.0
   numpy>=1.24
   scipy>=1.10
   matplotlib>=3.7
   Pillow>=10.0
   ```
3. Utworzyć opcjonalny `requirements-dev.txt`:
   ```
   -r requirements.txt
   pytest>=7.0
   opencv-python>=4.8
   ```
4. Zweryfikować wersje: `pip install -r requirements.txt` w czystym środowisku wirtualnym.
5. Rozważyć utworzenie `pyproject.toml` zamiast/obok `requirements.txt` dla lepszej standardyzacji.

**Kryterium zakończenia**: `pip install -r requirements.txt` instaluje wszystkie zależności bez błędów.

---

#### 2.2 Parametryzacja ścieżek i stałych

**Problem**: Ścieżki `quantum_dataset/` i parametry (`N=32`, `resolution=100`, `n_samples=10`) są wpisane na sztywno.

**Kroki**:
1. Utworzyć plik konfiguracyjny `src/physics/config.py`:
   ```python
   from dataclasses import dataclass, field
   from pathlib import Path

   @dataclass
   class GeneratorConfig:
       output_dir: Path = Path("quantum_dataset")
       cutoff: int = 32
       resolution: int = 100
       x_range: float = 5.0
       n_samples: int = 10
       cmap: str = "RdBu_r"
   ```
2. Zrefaktoryzować `generate.py`:
   - Przyjmować `GeneratorConfig` jako argument
   - Dodać interfejs CLI z `argparse`:
     ```
     python generate.py --n-samples 100 --resolution 200 --output-dir data/
     ```
3. Zrefaktoryzować `divider.py`:
   - Parametr `input_directory` zamiast zakodowanej ścieżki `quantum_dataset/clean/`
   - Parametr `square_size` z wartością domyślną
4. Usunąć wszystkie zakodowane na sztywno ścieżki z kodu źródłowego.

**Kryterium zakończenia**: `python generate.py --help` wyświetla dostępne opcje. Żadna ścieżka nie jest wpisana na sztywno.

---

#### 2.3 Ujednolicenie języka

**Kroki**:
1. Przetłumaczyć komentarze polskie w `generate.py` na angielski:
   - `# --- 1. STAN FOCKA ---` → `# --- 1. FOCK STATE ---`
   - `# Czysty` → `# Clean`
   - `f"Generowanie {n_samples} par stanów..."` → `f"Generating {n_samples} state pairs..."`
   - `print("Dane zapisano w folderze quantum_dataset/")` → `print(f"Data saved to {output_dir}")`
2. Przetłumaczyć komentarz w `gkp.py:10`:
   - `# liczba pików po obu stronach` → `# number of peaks on each side`
3. Przetłumaczyć komentarz w `gkp.py:21-22`:
   - `# operator przesunięcia` → `# displacement operator`
   - `# envelope Gaussowski` → `# Gaussian envelope`
4. Przejrzeć resztę kodu pod kątem pominiętych komentarzy polskich.

**Kryterium zakończenia**: `grep -rn "[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]" src/` nie zwraca żadnych wyników.

---

#### 2.4 Rozbudowa `README.md`

**Kroki**:
1. Dodać sekcje:
   - **Opis projektu** — rozszerzony opis celu i kontekstu (konkurs IDUB)
   - **Struktura projektu** — drzewo katalogów z opisem modułów
   - **Wymagania** — wersja Pythona, zależności
   - **Instalacja** — `git clone`, `pip install -r requirements.txt`
   - **Użycie** — przykłady generowania stanów i zbioru danych
   - **Zaimplementowane stany kwantowe** — tabela z listą stanów
   - **Kanały szumu** — opis dostępnych kanałów
   - **Autorzy** — lista członków zespołu
2. Dodać przykład szybkiego startu:
   ```python
   from src.physics.state.fock import FockState
   from src.physics.measurement.wigner import WignerMeasurement

   state = FockState(n=3, cutoff=32)
   wm = WignerMeasurement(x_max=5, resolution=100)
   W = wm.measure(state.density_matrix())
   ```
3. Dodać sekcję o generowaniu zbioru danych:
   ```bash
   python src/physics/generate.py --n-samples 100 --resolution 200
   ```

**Kryterium zakończenia**: README zawiera instrukcję instalacji i przykłady użycia. Nowy użytkownik może uruchomić projekt w 5 minut.

---

### Faza 3 — Jakość kodu

#### 3.1 Adnotacje typów

**Kroki**:
1. Dodać typy do klas stanów (`state/*.py`):
   ```python
   # Przykład dla FockState:
   class FockState(QuantumState):
       def __init__(self, n: int, cutoff: int) -> None:
           self.n: int = n
           self.cutoff: int = cutoff

       def ket(self) -> Qobj:
           return basis(self.cutoff, self.n)
   ```
2. Dodać typy do kanałów szumu:
   ```python
   class LossChannel(QuantumChannel):
       def __init__(self, cutoff: int, gamma: float) -> None: ...
       def apply(self, rho: Qobj) -> Qobj: ...
   ```
3. Dodać typy do `WignerMeasurement`:
   ```python
   def measure(self, rho: Qobj) -> np.ndarray: ...
   ```
4. Dodać typy do `MeasurementPipeline`:
   ```python
   from typing import Optional
   def __init__(self, measurement: WignerMeasurement, noise: Optional[QuantumChannel] = None) -> None: ...
   def run(self, state: QuantumState) -> np.ndarray: ...
   ```
5. Dodać typy do funkcji narzędziowych:
   ```python
   # validation.py
   def is_physical(rho: Qobj, tol: float = 1e-10) -> bool: ...

   # divider.py
   def divide_on_squares(img_name: str, x: int, output_directory: str = 'divided') -> None: ...
   ```
6. Uruchomić `mypy src/` i naprawić wykryte błędy typów.

**Kryterium zakończenia**: Wszystkie funkcje i metody publiczne mają adnotacje typów. `mypy src/` przechodzi bez błędów krytycznych.

---

#### 3.2 Docstringi

**Kroki**:
1. Dodać docstringi modułowe (na początku każdego pliku `.py`):
   ```python
   """Fock (number) state |n⟩ implementation.

   Fock states form the basis of the quantum harmonic oscillator
   Hilbert space and represent states with a definite photon number.
   """
   ```
2. Dodać docstringi klas — format Google Style:
   ```python
   class FockState(QuantumState):
       """Fock (number) state |n⟩.

       Args:
           n: Number of photons (n >= 0).
           cutoff: Hilbert space dimension (cutoff > n).
       """
   ```
3. Dodać docstringi metod publicznych:
   ```python
   def ket(self) -> Qobj:
       """Return the ket vector |n⟩ in the Fock basis.

       Returns:
           QuTiP Qobj representing the state vector.
       """
   ```
4. Dla stanów GKP i dwumianowego — dodać wzory matematyczne i referencje:
   ```python
   class GKPState(QuantumState):
       """Approximate Gottesman-Kitaev-Preskill (GKP) state.

       Constructs a finite-energy GKP state using Gaussian-weighted
       displaced vacuum states:
           |GKP⟩ ∝ Σ_s exp(-2π s² / δ²) D(s√(2π)) |0⟩

       Reference:
           Gottesman, Kitaev, Preskill, PRA 64, 012310 (2001).

       Args:
           cutoff: Hilbert space dimension.
           delta: Envelope width parameter (smaller = more ideal).
           grid_size: Number of displacement peaks on each side of origin.
       """
   ```
5. Przyjąć jednolity styl docstringów w całym projekcie (zalecany: Google Style).

**Kryterium zakończenia**: Każdy moduł, klasa i metoda publiczna mają docstringi. `pydocstyle src/` nie zgłasza krytycznych błędów.

---

#### 3.3 Walidacja danych wejściowych

**Kroki**:
1. `BinomialState.__init__()`:
   ```python
   def __init__(self, N: int, p: float, cutoff: int) -> None:
       if N < 0:
           raise ValueError(f"N must be non-negative, got {N}")
       if not 0 <= p <= 1:
           raise ValueError(f"p must be in [0, 1], got {p}")
       if cutoff <= N:
           raise ValueError(f"cutoff ({cutoff}) must be > N ({N})")
   ```
2. `GKPState.__init__()`:
   ```python
   if delta <= 0:
       raise ValueError(f"delta must be positive, got {delta}")
   if grid_size < 1:
       raise ValueError(f"grid_size must be >= 1, got {grid_size}")
   ```
3. `FockState.__init__()`:
   ```python
   if n < 0:
       raise ValueError(f"n must be non-negative, got {n}")
   if cutoff <= n:
       raise ValueError(f"cutoff ({cutoff}) must be > n ({n})")
   ```
4. `LossChannel.__init__()`:
   ```python
   if gamma < 0:
       raise ValueError(f"gamma must be non-negative, got {gamma}")
   ```
5. `MixtureChannel.__init__()`:
   ```python
   if not 0 <= p <= 1:
       raise ValueError(f"p must be in [0, 1], got {p}")
   ```
6. `ThermalState.__init__()`:
   ```python
   if n_th < 0:
       raise ValueError(f"n_th must be non-negative, got {n_th}")
   ```
7. `validation.py` — uczynić tolerancję konfigurowalną:
   ```python
   def is_physical(rho: Qobj, tol: float = 1e-10) -> bool:
   ```

**Kryterium zakończenia**: Wszystkie klasy odrzucają nieprawidłowe parametry z czytelnymi komunikatami `ValueError`.

---

#### 3.4 Obsługa błędów w skryptach

**Kroki**:
1. `divider.py` — dodać obsługę braku pliku:
   ```python
   def divide_on_squares(img_name: str, x: int, input_dir: str, output_dir: str = 'divided') -> int:
       img_path = os.path.join(input_dir, img_name)
       if not os.path.exists(img_path):
           raise FileNotFoundError(f"Image not found: {img_path}")
       if x <= 0:
           raise ValueError(f"Square size must be positive, got {x}")
       # ...
       return counter  # liczba zapisanych fragmentów
   ```
2. `generate.py` — dodać obsługę błędów zapisu:
   ```python
   try:
       plt.imsave(path, w_data, cmap=config.cmap)
   except OSError as e:
       print(f"Error saving {path}: {e}")
       continue
   ```
3. Dodać pasek postępu do `generate.py` (opcjonalnie z `tqdm` lub prosta wersja z `print`):
   ```python
   for i in range(n_samples):
       if (i + 1) % 10 == 0:
           print(f"  [{i+1}/{n_samples}]")
   ```

**Kryterium zakończenia**: Skrypty wyświetlają czytelne komunikaty błędów zamiast traceback Pythona.

---

#### 3.5 Referencje do literatury

**Kroki**:
1. `gkp.py` — dodać w docstringu klasy opis fizyczny i referencję:
   ```python
   """Approximate Gottesman-Kitaev-Preskill (GKP) state.

   GKP state encodes a logical qubit in a harmonic oscillator,
   protecting against small displacement errors in position (q)
   and momentum (p). The ideal state is an infinite comb of
   Dirac deltas spaced by 2√π. This implementation uses a
   finite-energy approximation with a Gaussian envelope of
   width δ:

       |GKP⟩ ∝ Σ_s exp(-2π s²/δ²) D(s√(2π)) |0⟩

   Reference:
       D. Gottesman, A. Kitaev, J. Preskill,
       "Encoding a qubit in an oscillator",
       Phys. Rev. A 64, 012310 (2001).
       https://doi.org/10.1103/PhysRevA.64.012310
   """
   ```
2. `binomial.py` — dodać referencję:
   ```
   Reference:
       M. H. Michael, M. Silveri, R. T. Brierley et al.,
       "New class of quantum error-correcting codes for a bosonic mode",
       Phys. Rev. X 6, 031006 (2016).
       https://doi.org/10.1103/PhysRevX.6.031006
   ```
3. `cat.py` — opcjonalna referencja do oryginalnej koncepcji kota Schrödingera i stanów cat w optyce kwantowej.
4. `loss.py` — dodać wzmiankę o kanale tłumienia amplitudy (amplitude damping channel) i równaniu Lindblada.

**Kryterium zakończenia**: Każdy niebanalny stan i kanał ma referencję do publikacji naukowej.

---

### Faza 4 — Testy

#### 4.1 Konfiguracja pytest

**Kroki**:
1. Zainstalować pytest: dodać do `requirements-dev.txt`
2. Utworzyć strukturę katalogów:
   ```
   tests/
   ├── __init__.py
   ├── conftest.py          — wspólne fixtures
   ├── test_states.py       — testy stanów
   ├── test_noise.py        — testy kanałów szumu
   ├── test_validation.py   — testy walidacji
   ├── test_measurement.py  — testy pomiarów
   └── test_pipeline.py     — testy integracyjne
   ```
3. Utworzyć `conftest.py` ze wspólnymi parametrami:
   ```python
   import pytest

   @pytest.fixture
   def cutoff():
       return 32

   @pytest.fixture
   def xvec():
       import numpy as np
       return np.linspace(-5, 5, 64)
   ```
4. Dodać sekcję `[tool.pytest.ini_options]` do przyszłego `pyproject.toml` lub `pytest.ini`.
5. Sprawdzić: `pytest tests/ -v` uruchamia się bez błędów (nawet jeśli testów jeszcze nie ma).

**Kryterium zakończenia**: `pytest` odkrywa i uruchamia testy z katalogu `tests/`.

---

#### 4.2 Testy jednostkowe stanów

Każdy stan kwantowy musi spełniać podstawowe warunki fizyczne (patrz sekcja 1a „Fizyczność macierzy gęstości"). Poniższe testy to weryfikują automatycznie:

**Przypadki testowe wspólne dla każdego stanu**:

| Test | Co sprawdzamy? | Dlaczego to ważne? | Sprawdzenie w kodzie |
|------|------|-------------|-------------|
| Normalizacja ketu | `⟨ψ\|ψ⟩ = 1` | Prawdopodobieństwa muszą sumować się do 1 | `abs(state.ket().norm() - 1) < 1e-10` |
| Ślad macierzy gęstości | `Tr(ρ) = 1` | j.w. ale dla macierzy gęstości | `abs(state.density_matrix().tr() - 1) < 1e-10` |
| Hermitowskość | `ρ = ρ†` | Wielkości obserwowalne muszą być rzeczywiste | `state.density_matrix().isherm == True` |
| Dodatnia półokreśloność | wartości własne ≥ 0 | Prawdopodobieństwa nie mogą być ujemne | `np.all(eigvals >= -1e-10)` |
| Fizyczność | Łączy 3 powyższe | Kompletna weryfikacja poprawności stanu | `is_physical(state.density_matrix()) == True` |

**Testy specyficzne dla stanów** — każdy stan ma unikalne cechy fizyczne, które można sprawdzić:

```python
# test_states.py

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
        a = destroy(cutoff)         # operator anihilacji
        n_hat = a.dag() * a         # operator liczby fotonów: n̂ = a†a
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
        assert abs(expect(n_hat, rho) - abs(alpha)**2) < 0.01

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
        # Sprawdzamy, że nieparzyste składowe Focka mają zerowe nakładanie
        for n in [1, 3, 5]:
            overlap = abs(basis(cutoff, n).dag() * psi)[0][0]
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
```

**Kryterium zakończenia**: Wszystkie testy stanów przechodzą. Pokrycie plików `state/*.py` > 80%.

---

#### 4.3 Testy kanałów szumu

Kanał szumu to operacja fizyczna, która zmienia stan kwantowy (patrz sekcja 1a „Kanały szumu kwantowego"). Kluczowa zasada: **kanał szumu nie może złamać praw fizyki** — wynik musi nadal być poprawną macierzą gęstości.

**Przypadki testowe**:

```python
# test_noise.py

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
```

**Kryterium zakończenia**: Wszystkie testy kanałów przechodzą. Kanały zachowują fizyczność stanów.

---

#### 4.4 Testy walidacji

Funkcja `is_physical()` sprawdza, czy macierz gęstości jest poprawna fizycznie (patrz sekcja 1a „Fizyczność macierzy gęstości"). Testujemy ją na przykładach poprawnych i celowo uszkodzonych macierzy:

**Przypadki testowe**:

```python
# test_validation.py

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
```

**Kryterium zakończenia**: `is_physical()` poprawnie klasyfikuje stany fizyczne i niefizyczne.

---

#### 4.5 Testy integracyjne pipeline

Testy integracyjne sprawdzają, czy cały potok (pipeline) działa od początku do końca: tworzenie stanu → (opcjonalnie) zastosowanie szumu → pomiar funkcją Wignera → tablica 2D z wartościami.

**Przypadki testowe**:

```python
# test_pipeline.py

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
```

**Kryterium zakończenia**: Pełny pipeline (stan → szum → pomiar) działa dla wszystkich typów stanów. `pytest tests/ -v` — 100% passed.

---

### Faza 5 — Rozwój funkcjonalności

#### 5.1 Nowe kanały szumu

Obecnie projekt ma 2 kanały szumu. W rzeczywistości światło kwantowe podlega wielu rodzajom niedoskonałości. Poniżej 3 nowe kanały do zaimplementowania:

**Kanał defazowania (dephasing)** — niszczenie informacji o fazie (kącie w przestrzeni fazowej), bez utraty fotonów. Analogia: światło przechodzące przez ośrodek o losowo zmieniającym się współczynniku załamania.
1. Utworzyć plik `src/physics/noise/dephasing.py`:
   ```python
   class DephasingChannel(QuantumChannel):
       """Phase damping channel using Lindblad master equation.

       Collapse operator: √γ * n̂ (number operator).
       """
       def __init__(self, cutoff: int, gamma: float) -> None: ...
       def apply(self, rho: Qobj) -> Qobj: ...
   ```
2. Operator kolapsu: `c_ops = [sqrt(gamma) * a.dag() * a]` — operator liczby fotonów n̂ = a†a powoduje utratę koherencji fazowej proporcjonalnie do liczby fotonów.
3. Użyć `mesolve()` analogicznie jak w `LossChannel`.

**Kanał wzmocnienia (amplification)** — dodawanie fotonów do układu (odwrotność strat). Analogia: wzmacniacz optyczny, który dodaje fotony, ale wprowadza szum kwantowy.
1. Utworzyć plik `src/physics/noise/amplification.py`:
   ```python
   class AmplificationChannel(QuantumChannel):
       """Quantum amplification channel.

       Collapse operator: √γ * a† (creation operator).
       """
   ```
2. Operator kolapsu: `c_ops = [sqrt(gamma) * a.dag()]` — operator kreacji a† dodaje fotony.

**Kanał depolaryzacji** — z prawdopodobieństwem p stan zostaje zastąpiony „białym szumem" (stanem maksymalnie mieszanym I/d, gdzie d to wymiar przestrzeni). Analogia: detektor, który z pewnym prawdopodobieństwem daje całkowicie losowy wynik.
1. Utworzyć `src/physics/noise/depolarizing.py`:
   ```python
   class DepolarizingChannel(QuantumChannel):
       """Depolarizing channel: ρ → (1-p)ρ + p·I/d."""
   ```

**Kroki wspólne**:
4. Dodać nowe kanały do `__init__.py` i do eksportów.
5. Napisać testy jednostkowe dla każdego kanału (analogicznie do fazy 4.3).
6. Zweryfikować zachowanie fizyczności na różnych stanach wejściowych.

**Kryterium zakończenia**: Minimum 3 nowe kanały szumu, każdy z testami i dokumentacją.

---

#### 5.2 Generowanie stanów mieszanych

**Problem**: Obecnie generator tworzy tylko czyste stany (poza termicznym). Dla realistycznego zbioru danych ML potrzebne są stany mieszane.

**Kroki**:
1. Rozszerzyć `generate.py` o generowanie stanów z szumem:
   - Fock + kanał strat → stan mieszany
   - Koherentny + defazowanie → stan mieszany
   - Kot + straty → zdekohowany kot
2. Dodać osobne podkatalogi wyjściowe:
   ```
   quantum_dataset/
   ├── clean/         — stany czyste
   ├── loss_0.1/      — po kanale strat γ=0.1
   ├── loss_0.3/      — po kanale strat γ=0.3
   └── dephasing_0.2/ — po defazowaniu γ=0.2
   ```
3. Dodać etykiety do nazw plików lub plik CSV z metadanymi:
   ```csv
   filename,state_type,noise_type,noise_param,n_photons,alpha,p
   fock_n3_loss0.1_id0.png,fock,loss,0.1,3,,
   ```
4. Metadane będą kluczowe dla późniejszego modułu ML.

**Kryterium zakończenia**: Generator tworzy zbiór danych z wariantami czystymi i zaszumionymi. Metadane zapisane w CSV.

---

#### 5.3 Konfiguracja CI/CD

**Kroki**:
1. Utworzyć plik `.github/workflows/tests.yml`:
   ```yaml
   name: Tests
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: ["3.10", "3.11", "3.12"]
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: ${{ matrix.python-version }}
         - run: pip install -r requirements-dev.txt
         - run: pytest tests/ -v --tb=short
   ```
2. Dodać badge statusu testów do `README.md`.
3. Opcjonalnie dodać linting (`ruff` lub `flake8`) i sprawdzanie typów (`mypy`) do pipeline.
4. Rozważyć konfigurację pre-commit hooks (`.pre-commit-config.yaml`).

**Kryterium zakończenia**: Pull requesty automatycznie uruchamiają testy. Badge w README pokazuje status.

---

#### 5.4 Moduł ML — klasyfikator stanów

**Zakres**: To jest główny cel projektu i największe otwarte zadanie.

**Kroki wstępne**:
1. Określić architekturę sieci:
   - CNN (np. ResNet, EfficientNet) do klasyfikacji obrazów Wignera
   - Autoencoder do rekonstrukcji stanów
2. Wybrać framework ML: PyTorch lub TensorFlow/Keras.
3. Zaprojektować strukturę katalogów:
   ```
   src/ml/
   ├── __init__.py
   ├── dataset.py        — DataLoader dla obrazów Wignera
   ├── models/
   │   ├── classifier.py  — CNN do klasyfikacji stanów
   │   └── reconstructor.py — autoencoder do rekonstrukcji
   ├── train.py           — skrypt treningowy
   └── evaluate.py        — ewaluacja modeli
   ```
4. Zdefiniować zadania ML:
   - **Klasyfikacja**: obraz Wignera → typ stanu (Fock/koherentny/kot/GKP/dwumianowy/termiczny)
   - **Rekonstrukcja**: zaszumiony obraz Wignera → czysty obraz Wignera
   - **Estymacja parametrów**: obraz Wignera → parametry stanu (n, α, p)
5. Wygenerować odpowiednio duży zbiór danych (minimum 1000 obrazów na klasę).
6. Zaimplementować, wytrenować i zewaluować modele.

**Kryterium zakończenia**: Działający klasyfikator z dokładnością > 90% na zbiorze testowym.

---

#### 5.5 Optymalizacja wydajności

**Problem**: `GKPState.ket()` tworzy `2 * grid_size + 1` operatorów przesunięcia przy każdym wywołaniu. Dla dużych `cutoff` to może być wolne.

**Kroki**:
1. Zprofilować generowanie stanów:
   ```python
   import cProfile
   cProfile.run('GKPState(cutoff=64, delta=0.3).ket()')
   ```
2. Jeśli `displace()` dominuje czas — dodać cache:
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=128)
   def _displacement_op(cutoff: int, alpha: complex) -> Qobj:
       return displace(cutoff, alpha)
   ```
3. Rozważyć wektoryzację: zamiast pętli po `s`, obliczyć wagi jako wektor NumPy i zastosować sumę ważoną operatorów.
4. Dla generatora danych — użyć `multiprocessing` do równoległego generowania obrazów:
   ```python
   from multiprocessing import Pool
   with Pool(4) as pool:
       pool.map(generate_single_state, state_params)
   ```
5. Zmierzyć przyspieszenie i udokumentować.

**Kryterium zakończenia**: Generowanie zbioru 1000 obrazów zajmuje < 10 minut (do zmierzenia po implementacji).

---

---

## 6. Harmonogram i zależności między fazami

```
Faza 1 (Błędy krytyczne)  ──→  Faza 2 (Infrastruktura)  ──→  Faza 3 (Jakość kodu)
         ~4 godz.                      ~5 godz.                     ~11 godz.
                                          │
                                          ▼
                                  Faza 4 (Testy)  ──→  Faza 5 (Rozwój)
                                     ~10 godz.           ~otwarte
```

**Zależności**:
- Faza 2 wymaga ukończenia Fazy 1 (pakiety muszą działać przed konfiguracją CLI)
- Faza 4 wymaga ukończenia Fazy 1 i częściowo Fazy 3 (walidacja potrzebna do testów)
- Faza 3 i 4 mogą być realizowane częściowo równolegle
- Faza 5 wymaga stabilnej bazy z Faz 1–4

**Łączny szacowany nakład**: ~30 godzin roboczych (Fazy 1–4) + otwarte (Faza 5, głównie moduł ML)

---

## 7. Podsumowanie

Projekt ma **solidne fundamenty architektoniczne** — przejrzysty podział na moduły, poprawne użycie klas abstrakcyjnych i trafny dobór bibliotek. Większość implementacji fizycznych jest poprawna.

Główne braki to: **brak struktury pakietu** (`__init__.py`), **brak testów**, **brak dokumentacji** oraz **kilka błędów matematycznych** do zweryfikowania. Brakuje też części ML, która jest kluczowym celem projektu (klasyfikacja i rekonstrukcja stanów).

**Zalecana kolejność prac**: Faza 1 → Faza 2 → Faza 4 → Faza 3 → Faza 5.

Szacowany łączny nakład pracy na fazy 1–4: **~30 godzin roboczych**.
