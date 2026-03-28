# Plan rozwoju repozytorium IDUB_2026

**Cel**: Doprowadzenie repozytorium do realizacji wszystkich celów projektu IDUB opisanych w README.md.

**Stan wyjściowy**: Zaimplementowane 7 stanów kwantowych, 2 kanały szumu, pomiar Wignera i pipeline. Brak kodu ML, testów, dokumentacji, struktury pakietu.

**Harmonogram projektu**: 36 tygodni, 4 osoby.

---

## Bilans: co jest, czego brakuje

| Cel projektu | Status w repo | Co trzeba zrobić |
|---|---|---|
| Symulatory 7 stanów kwantowych | **90%** — wszystkie zaimplementowane, błędy do naprawy | Naprawy z raportu audytu (Faza 1) |
| Modele szumu (straty, gauss, mieszanie) | **30%** — 2 z 5+ kanałów | Dodać: defazowanie, depolaryzację, szum gaussowski, wzmocnienie |
| Baza danych >10 000 stanów | **10%** — generator istnieje, ale sampler ma 2/7 stanów | Rozszerzyć sampler, dodać augmentację, train/val/test split |
| Klasyfikator CNN (7 klas, Wigner/Husimi Q) | **0%** | Pełna implementacja |
| Funkcja Husimi Q | **0%** | Implementacja analogiczna do Wignera |
| cGAN z warstwami kwantowymi | **0%** | Pełna implementacja z niestandardowymi warstwami |
| Warstwy macierzy gęstości (hermitowska, ślad=1) | **0%** | Implementacja jako custom layers PyTorch/TF |
| Porównanie z MLE i APG | **0%** | Implementacja metod klasycznych |
| Analiza odporności na szum | **0%** | Systematyczne eksperymenty |
| Testy automatyczne | **0%** | pytest, testy stanów/kanałów/pipeline/ML |
| Dokumentacja kodu | **10%** | Docstringi, README, tutoriale |
| Narzędzia wizualizacyjne | **5%** — tylko zapis PNG | Interaktywne wykresy Wignera, macierzy gęstości |
| Publikacja naukowa | **0%** | Tekst, wykresy, analiza |

---

## Etap 0: Naprawy infrastrukturalne (Tydzień 1) — 1 osoba

Realizacja zadań z raportów audytu (`raport.tex`, `raport_quantum_statesDB.tex`). Bez tego dalszy rozwój jest zablokowany.

### 0.1 Struktura pakietu
- [ ] Dodać `__init__.py` do `src/`, `src/physics/`, `src/physics/state/`, `src/physics/noise/`, `src/physics/measurement/`, `src/dataset/`
- [ ] Ujednolicić importy do względnych (`from .base import ...`)
- [ ] Zmienić nazwę `binomal.py` → `binomial.py`
- [ ] Dodać dziedziczenie `QuantumChannel` w `LossChannel` i `MixtureChannel`
- [ ] Naprawić błąd B2 w `generate.py` (stan dwumianowy)

### 0.2 Konfiguracja projektu
- [ ] Utworzyć `requirements.txt`: qutip, numpy, scipy, matplotlib, pillow
- [ ] Utworzyć `requirements-ml.txt`: torch (lub tensorflow), torchvision, scikit-learn, tqdm, tensorboard
- [ ] Utworzyć `requirements-dev.txt`: pytest, mypy, ruff
- [ ] Dodać `.github/workflows/tests.yml` (CI)
- [ ] Dodać `pyproject.toml` z konfiguracją pytest i mypy

### 0.3 Porządki
- [ ] Przetłumaczyć komentarze polskie na angielski
- [ ] Dodać type hints i docstringi do istniejących klas
- [ ] Dodać walidację parametrów w konstruktorach
- [ ] Usunąć lub oznaczyć jako przestarzały stary `src/physics/generate.py`

**Kryterium**: `python -c "from src.physics import FockState"` działa. `pytest` uruchamia się. CI przechodzi.

---

## Etap 1: Rozbudowa symulacji fizycznych (Tygodnie 1–3) — 2 osoby

Doprowadzenie warstwy fizycznej do kompletności wymaganej przez cele projektu.

### 1.1 Nowe kanały szumu
Projekt wymaga badania odporności na „straty fotonów, szum gaussowski, mieszanie stanów". Obecne 2 kanały to za mało.

- [ ] **DephasingChannel** — operator kolapsu: √γ·n̂ (utrata fazy bez utraty fotonów)
- [ ] **DepolarizingChannel** — ρ → (1-p)ρ + p·I/d (całkowita utrata informacji z prawdopodobieństwem p)
- [ ] **AmplificationChannel** — operator kolapsu: √γ·a† (spontaniczne wzmocnienie)
- [ ] **GaussianNoiseChannel** — dodanie szumu gaussowskiego do kwadratur (wymagane explicite w opisie projektu)
- [ ] **ThermalNoiseChannel** — dodanie fotonów termicznych (mieszanie ze stanem termicznym)

Pliki: `src/physics/noise/dephasing.py`, `depolarizing.py`, `amplification.py`, `gaussian.py`, `thermal_noise.py`

### 1.2 Funkcja Husimi Q
Opis projektu wymaga klasyfikacji „na podstawie funkcji Husimi Q lub Wignera". Obecna implementacja ma tylko Wignera.

- [ ] **HusimiMeasurement** w `src/physics/measurement/husimi.py`

Funkcja Husimi Q: Q(α) = ⟨α|ρ|α⟩ / π — zawsze nieujemna, prostsza do interpretacji niż Wigner.

```python
class HusimiMeasurement:
    def __init__(self, x_max=5, resolution=64):
        self.xvec = np.linspace(-x_max, x_max, resolution)

    def measure(self, rho):
        # QuTiP: qutip.visualization.hinton() lub ręcznie:
        # Q(alpha) = <alpha|rho|alpha> / pi
        from qutip import coherent
        Q = np.zeros((len(self.xvec), len(self.xvec)))
        cutoff = rho.shape[0]
        for i, x in enumerate(self.xvec):
            for j, p in enumerate(self.xvec):
                alpha = (x + 1j * p) / np.sqrt(2)
                coh = coherent(cutoff, alpha)
                Q[i, j] = np.real((coh.dag() * rho * coh)[0, 0]) / np.pi
        return Q
```

Uwaga: powyższa implementacja jest wolna (O(res²·cutoff)). Alternatywnie użyć `qutip.wigner.qfunc()`.

### 1.3 Testy warstwy fizycznej
- [ ] Testy normalizacji i fizyczności wszystkich 7 stanów
- [ ] Testy kanałów szumu: zachowanie fizyczności, redukcja fotonów (straty), zachowanie fotonów (defazowanie)
- [ ] Testy Husimi Q: nieujemność, normalizacja ∫Q dα = 1
- [ ] Test pipeline: stan → szum → {Wigner, Husimi} → tablica 2D

**Kryterium**: Wszystkie testy przechodzą. 7 stanów × 7 kanałów × 2 pomiary = 98 kombinacji testowych.

---

## Etap 2: Generator danych ML-ready (Tygodnie 3–5) — 1–2 osoby

Cel projektu: „Generacja pierwszych zbiorów danych uczących (>10 000 stanów)".

### 2.1 Rozszerzony sampler
- [ ] `full_sampler()` z wszystkimi 7 stanami i losowymi parametrami
- [ ] `LABEL_MAP` z mapowaniem numer → nazwa
- [ ] Konfigurowalny zakres parametrów (np. alpha ∈ [0.5, 3.5], n ∈ [0, 7])
- [ ] Zbalansowane próbkowanie (równa liczba próbek na klasę)

### 2.2 Rozszerzony generator
- [ ] Generowanie z konfigurowalnymi kanałami szumu i ich parametrami
- [ ] Wiele poziomów szumu na stan: γ ∈ {0, 0.05, 0.1, 0.2, 0.5}
- [ ] Dwa typy pomiarów: Wigner i Husimi Q
- [ ] Zapis metadanych: typ stanu, parametry, typ szumu, siła szumu
- [ ] Pasek postępu (tqdm)
- [ ] Odtwarzalność (random seed)

### 2.3 Format danych i podział
- [ ] Zapis do `.npz` z polami: X (dane), y (etykiety), metadata, label_map
- [ ] Automatyczny podział: train (70%) / val (15%) / test (15%)
- [ ] Normalizacja danych (opcjonalna: min-max, z-score)
- [ ] Interfejs CLI: `python -m scripts.generate_dataset --n-samples 10000 --noise loss --gamma 0.1 --measurement wigner --seed 42 --output data/`

### 2.4 PyTorch Dataset
- [ ] `src/dataset/quantum_dataset.py` — klasa `QuantumStateDataset(torch.utils.data.Dataset)`
- [ ] Ładowanie z `.npz`, obsługa transformacji, lazy loading dla dużych zbiorów
- [ ] DataLoader z batch_size, shuffle, num_workers

**Kryterium**: Wygenerowanie zbioru 10 000+ stanów (7 klas × ~1400 próbek, 5 poziomów szumu) zajmuje < 30 min. DataLoader zwraca batch tensora o kształcie (B, 1, 64, 64).

---

## Etap 3: Klasyfikator CNN (Tygodnie 5–12) — 2 osoby

Cel projektu: „Klasyfikator na bazie CNN do identyfikacji 7 klas stanów kwantowych".

### 3.1 Architektura sieci

Proponowana struktura katalogów:
```
src/ml/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── classifier.py      — CNN klasyfikator
│   ├── layers.py           — niestandardowe warstwy kwantowe
│   └── reconstructor.py    — cGAN rekonstruktor
├── training/
│   ├── __init__.py
│   ├── trainer.py          — pętla treningowa
│   ├── losses.py           — funkcje kosztu
│   └── metrics.py          — metryki (accuracy, F1, confusion matrix)
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py        — ewaluacja modelu
│   └── noise_analysis.py   — analiza odporności na szum
└── visualization/
    ├── __init__.py
    ├── wigner_plot.py       — wizualizacja Wignera/Husimi
    ├── training_plot.py     — krzywe uczenia
    └── comparison_plot.py   — wykresy porównawcze metod
```

- [ ] **Klasyfikator CNN** (`models/classifier.py`):
  - Wejście: (1, 64, 64) — funkcja Wignera lub Husimi Q (jednokanałowa)
  - Architektura bazowa: 4 bloki konwolucyjne (Conv2d → BatchNorm → ReLU → MaxPool) + 2 warstwy FC + Softmax
  - Wariant zaawansowany: adaptacja ResNet-18 lub EfficientNet-B0 (pretrained na ImageNet, fine-tuned)
  - Wyjście: 7 klas

- [ ] **Pętla treningowa** (`training/trainer.py`):
  - Cross-entropy loss
  - Optimizer: Adam z learning rate scheduler (CosineAnnealing lub ReduceOnPlateau)
  - Walidacja krzyżowa (k-fold, k=5)
  - Early stopping
  - Logowanie do TensorBoard
  - Zapis najlepszego modelu (checkpoint)

### 3.2 Optymalizacja hiperparametrów
- [ ] Grid search lub Optuna: learning rate, batch size, dropout, architektura
- [ ] Porównanie: Wigner vs Husimi Q jako dane wejściowe
- [ ] Porównanie: różne rozdzielczości (32×32, 64×64, 128×128)

### 3.3 Metryki i ewaluacja
- [ ] Macierz pomyłek (confusion matrix) dla 7 klas
- [ ] Krzywe ROC (one-vs-rest) i AUC
- [ ] Precision, recall, F1-score per klasa
- [ ] Accuracy ogólna i per-class

### 3.4 Analiza odporności na szum
Cel projektu: „Badanie odporności na różne typy szumu eksperymentalnego".

- [ ] Trenowanie na czystych danych → testowanie na zaszumionych (różne γ)
- [ ] Trenowanie na zaszumionych danych → testowanie na zaszumionych
- [ ] Wykresy: accuracy vs poziom szumu, per typ szumu (straty, defazowanie, depolaryzacja, gauss)
- [ ] Wykresy: accuracy vs typ stanu × typ szumu (heatmapa)
- [ ] Analiza: które stany są najtrudniejsze do sklasyfikowania przy szumie

**Kryterium**: Accuracy > 95% na czystych danych, > 85% przy umiarkowanym szumie (γ=0.1).

---

## Etap 4: Rekonstrukcja — cGAN z warstwami kwantowymi (Tygodnie 12–22) — 2 osoby

Cel projektu: „Warunkowa generatywna sieć przeciwstawna dla tomografii z niestandardowymi warstwami kwantowymi".

### 4.1 Niestandardowe warstwy kwantowe

To jest najbardziej nowatorska część projektu. Sieć generująca musi produkować poprawne fizycznie macierze gęstości.

- [ ] **DensityMatrixLayer** (`models/layers.py`):
  - Wejście: wektor latentny z
  - Wyjście: macierz gęstości ρ spełniająca: hermitowska, Tr(ρ)=1, dodatnio półokreślona
  - Implementacja: ρ = T†T / Tr(T†T), gdzie T to dowolna macierz (Cholesky-like parametrization)
  - Gradient musi przechodzić przez dekompozycję

- [ ] **ExpectationValueLayer** (`models/layers.py`):
  - Wejście: macierz gęstości ρ
  - Wyjście: wartość oczekiwana obserwabli ⟨O⟩ = Tr(ρO)
  - Reguła Borna — łączy macierz gęstości z wynikami pomiarów
  - Obserwable jako parametry warstwy (trainable lub fixed)

- [ ] **WignerLayer** (`models/layers.py`):
  - Wejście: macierz gęstości ρ
  - Wyjście: funkcja Wignera W(q,p) jako obraz 2D
  - Różniczkowalna implementacja transformaty Wignera

### 4.2 Architektura cGAN

- [ ] **Generator** (`models/reconstructor.py`):
  - Wejście: zaszumiona funkcja Wignera (1, 64, 64) + etykieta klasy (warunkowanie)
  - Architektura: Encoder-Decoder (U-Net style) z DensityMatrixLayer na wyjściu
  - Wyjście: zrekonstruowana macierz gęstości → czysta funkcja Wignera

- [ ] **Dyskryminator** (`models/reconstructor.py`):
  - Wejście: funkcja Wignera (prawdziwa lub wygenerowana) + etykieta
  - Architektura: PatchGAN lub standard CNN
  - Wyjście: skalar (prawdziwy/fałszywy)

- [ ] **Funkcje kosztu**:
  - Adversarial loss (GAN)
  - L1/L2 reconstruction loss (pixel-wise)
  - Fidelity loss: F(ρ_pred, ρ_true) = (Tr√(√ρ_true · ρ_pred · √ρ_true))²
  - Trace distance: T(ρ_pred, ρ_true) = ½ Tr|ρ_pred - ρ_true|

### 4.3 Trening cGAN
- [ ] Trening antagonistyczny z balansowaniem G/D
- [ ] Techniki stabilizacji: spectral normalization, gradient penalty
- [ ] Logowanie do TensorBoard: loss G/D, wierność rekonstrukcji, wizualizacje
- [ ] Checkpointy i early stopping na podstawie wierności

### 4.4 Metryki rekonstrukcji
- [ ] **Wierność** (fidelity) F(ρ_pred, ρ_true) — cel: > 0.99 dla stanów czystych
- [ ] **Trace distance** — miara odległości między stanami
- [ ] **Hilbert-Schmidt distance** — nawiązanie do pracy Trávníček, Bartkiewicz et al. (PRL 2019)
- [ ] Porównanie: cGAN vs MLE vs APG

**Kryterium**: Wierność > 0.95 dla stanów czystych z umiarkowanym szumem. Przyspieszenie ≥ 10× vs MLE.

---

## Etap 5: Metody klasyczne — baseline (Tygodnie 10–14) — 1 osoba

Cel projektu: „Analiza porównawcza z klasycznymi metodami estymacji".

### 5.1 Maximum Likelihood Estimation (MLE)
- [ ] `src/ml/baselines/mle.py`
- [ ] Implementacja iteracyjnej estymacji ρ z danych pomiarowych
- [ ] Wymuszenie fizyczności: ρ hermitowska, Tr(ρ)=1, ρ≥0
- [ ] Metryki: wierność, czas obliczeń

### 5.2 Accelerated Projected Gradient (APG)
- [ ] `src/ml/baselines/apg.py`
- [ ] Szybsza konwergencja niż MLE (metoda Nesterowa)
- [ ] Projekcja na zbiór macierzy gęstości

### 5.3 Framework porównawczy
- [ ] `src/ml/evaluation/benchmark.py`
- [ ] Wspólne metryki: wierność, trace distance, czas
- [ ] Wspólne dane testowe: te same stany, te same poziomy szumu
- [ ] Tabele i wykresy porównawcze

**Kryterium**: Tabela porównawcza CNN/cGAN vs MLE vs APG dla wszystkich 7 stanów × 5 poziomów szumu.

---

## Etap 6: Narzędzia wizualizacyjne (Tygodnie 14–16) — 1 osoba

Cel projektu: „Interaktywne narzędzia do analizy funkcji Wignera, macierzy gęstości i map aktywacji".

### 6.1 Moduł wizualizacji
- [ ] `src/ml/visualization/wigner_plot.py` — wykresy Wignera i Husimi Q z matplotlib/plotly
- [ ] `src/ml/visualization/density_matrix_plot.py` — wizualizacja macierzy gęstości (Hinton diagram, 3D bar)
- [ ] `src/ml/visualization/training_plot.py` — krzywe uczenia, loss, accuracy
- [ ] `src/ml/visualization/comparison_plot.py` — wykresy porównawcze metod

### 6.2 Interaktywny notebook
- [ ] Jupyter notebook z przykładami użycia wszystkich narzędzi
- [ ] Widgety: wybór stanu, parametrów, szumu → natychmiastowa wizualizacja
- [ ] Demo: generowanie stanu → klasyfikacja → rekonstrukcja → porównanie metod

**Kryterium**: Notebook z demonstracją pełnego potoku: generowanie → klasyfikacja → rekonstrukcja → porównanie.

---

## Etap 7: Eksperymenty i analiza (Tygodnie 22–30) — 4 osoby

### 7.1 Eksperymenty klasyfikacyjne
- [ ] Systematyczne eksperymenty: 7 stanów × 5 szumów × 5 poziomów γ × 2 pomiary (Wigner, Husimi)
- [ ] Walidacja krzyżowa 5-fold na każdej konfiguracji

### 7.2 Eksperymenty rekonstrukcyjne
- [ ] Rekonstrukcja 7 stanów czystych → wierność
- [ ] Rekonstrukcja stanów mieszanych → wierność
- [ ] Zależność wierności od poziomu szumu
- [ ] Zależność wierności od liczby danych treningowych

### 7.3 Analiza porównawcza
- [ ] CNN vs MLE vs APG: czas i dokładność
- [ ] cGAN vs MLE vs APG: wierność i czas
- [ ] Nawiązanie do wyników Ahmed et al. (98% accuracy, F>0.99)

### 7.4 Wyniki do publikacji
- [ ] Tabele z wynikami (LaTeX)
- [ ] Wykresy publikacyjne (matplotlib, format wektorowy)
- [ ] Analiza statystyczna (przedziały ufności, testy istotności)

---

## Etap 8: Dokumentacja i publikacja (Tygodnie 30–36) — 4 osoby

### 8.1 Dokumentacja kodu
- [ ] Docstringi we wszystkich modułach
- [ ] README z instrukcją instalacji, użycia, przykładami
- [ ] Tutoriale w Jupyter notebooks
- [ ] API reference (opcjonalnie: Sphinx)

### 8.2 Publikacja naukowa
- [ ] Manuskrypt: wprowadzenie, metody, wyniki, dyskusja
- [ ] Target: Physical Review A / Quantum Science and Technology / Scientific Reports
- [ ] Referencje do prac Bartkiewicza i Ahmed et al.

### 8.3 Materiały konferencyjne
- [ ] Prezentacja (slajdy)
- [ ] Poster

---

## Harmonogram — mapowanie na tygodnie

```
Tydzień:  1    3    5    8   10   12   14   16   18   22   26   30   36
          │    │    │    │    │    │    │    │    │    │    │    │    │
Etap 0:   ████                                                        Naprawy
Etap 1:   ████████                                                    Fizyka
Etap 2:        ████████                                               Dane
Etap 3:             ████████████████████                              CNN
Etap 4:                       ████████████████████████                cGAN
Etap 5:                  ████████████                                 Baseline
Etap 6:                            ████████                           Wizualizacja
Etap 7:                                      ████████████████████     Eksperymenty
Etap 8:                                                ████████████   Dokumentacja
```

## Podział na osoby

| Osoba | Główne odpowiedzialności | Etapy |
|---|---|---|
| **Osoba 1** | Architektura CNN, trening, analiza odporności | 3, 7.1 |
| **Osoba 2** | Architektura cGAN, warstwy kwantowe | 4, 7.2 |
| **Osoba 3** | Generator danych, augmentacja, baseline MLE/APG | 2, 5, 7.3 |
| **Osoba 4** | Wizualizacja, testy, dokumentacja, eksperymenty | 1, 6, 7.4, 8 |
| **Wszyscy** | Naprawy infrastrukturalne, publikacja | 0, 8.2 |

## Szacowany nakład pracy

| Etap | Godziny | Osoby |
|---|---|---|
| 0. Naprawy | 10 | 1 |
| 1. Fizyka | 25 | 2 |
| 2. Dane | 20 | 1–2 |
| 3. CNN | 40 | 2 |
| 4. cGAN | 60 | 2 |
| 5. Baseline | 20 | 1 |
| 6. Wizualizacja | 15 | 1 |
| 7. Eksperymenty | 35 | 4 |
| 8. Dokumentacja | 30 | 4 |
| **Łącznie** | **~255 godzin** | |
