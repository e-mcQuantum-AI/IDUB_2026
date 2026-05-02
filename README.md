# IDUB_2026

**Głębokie uczenie do klasyfikacji i rekonstrukcji kwantowych stanów optycznych**

Projekt realizowany w ramach konkursu *Inicjatywa Doskonałości — Uczelnia Badawcza* (IDUB).

- **Opiekun naukowy**: dr hab. Karol Bartkiewicz, prof. UAM
- **Instytucja**: Instytut Spintroniki i Informacji Kwantowej, Wydział Fizyki, Uniwersytet im. Adama Mickiewicza w Poznaniu
- **Zespół**: 4 osoby (studenci fizyki i informatyki)
- **Czas realizacji**: 36 tygodni

---

## 1. Uzasadnienie merytoryczne

Kwantowa tomografia stanów (Quantum State Tomography, QST) stanowi fundamentalną procedurę w informacji kwantowej, umożliwiającą pełną charakteryzację stanu kwantowego na podstawie danych pomiarowych. Tradycyjne metody, takie jak estymacja największej wiarygodności, wymagają znacznych zasobów obliczeniowych oraz dużej liczby pomiarów, co ogranicza praktyczne zastosowania technologii kwantowych.

Dr hab. Karol Bartkiewicz wraz z zespołem z Instytutu Spintroniki i Informacji Kwantowej UAM prowadzi pionierskie badania w obszarze kwantowego uczenia maszynowego. W ostatnich pracach opublikowanych w *Scientific Reports* (2020, 2023) oraz *Physical Review Letters* (2019) zademonstrowano zastosowanie sieci neuronowych do efektywnej tomografii kwantowej oraz detekcji splątania. W szczególności, w pracy z 2024 roku (Tulewicz, Bartkiewicz, Miranowicz, Nori) wykazano, że łączenie pomiarów wielokopijnych ze sztucznymi sieciami neuronowymi umożliwia redukcję wymagań pomiarowych o 67% w porównaniu z pełną tomografią stanów.

Projekt opiera się również na pracy Ahmed i współpracowników (*Physical Review Research*, 2021), w której zademonstrowano, że sieci neuronowe mogą osiągnąć dokładność klasyfikacji przekraczającą 98% oraz wierność rekonstrukcji powyżej 0,99 dla czystych stanów kwantowych, przy jednoczesnym przyspieszeniu procesu o 1–2 rzędy wielkości w porównaniu z metodami iteracyjnymi.

Proponowany projekt ma na celu implementację i walidację metod głębokiego uczenia dla optycznych stanów kwantowych, obejmujących stany Focka, koherentne, termiczne, stany kota Schrödingera, binomialne oraz Gottesmana-Kitaeva-Preskilla, ze szczególnym uwzględnieniem odporności metod w obecności szumu eksperymentalnego.

## 2. Cele projektu

**Cel główny**: Opracowanie, implementacja i walidacja metod głębokiego uczenia maszynowego do klasyfikacji i rekonstrukcji optycznych stanów kwantowych w kontekście osiągnięć zespołu prof. Bartkiewicza w dziedzinie kwantowego uczenia maszynowego.

**Cele szczegółowe**:

1. Implementacja klasyfikatora na bazie konwolucyjnej sieci neuronowej do identyfikacji siedmiu klas stanów kwantowych na podstawie funkcji Husimi Q lub Wignera
2. Rozwój architektury warunkowej generatywnej sieci przeciwstawnej (cGAN) dla tomografii z niestandardowymi warstwami kwantowymi (macierz gęstości, wartości oczekiwane)
3. Analiza porównawcza z klasycznymi metodami estymacji oraz podejściem wielokopijnym zespołu prof. Bartkiewicza
4. Badanie odporności na różne typy szumu eksperymentalnego
5. Interpretacja decyzji sieci metodą Grad-CAM w celu optymalizacji strategii pomiarowych

## 3. Struktura repozytorium

```
src/
├── physics/
│   ├── state/       — implementacje stanów kwantowych (Fock, koherentny, kot, GKP, dwumianowy, termiczny, próżnia)
│   ├── noise/       — kanały szumu (straty fotonowe, mieszanina)
│   ├── measurement/ — pomiar funkcją Wignera + pipeline
│   ├── hilbert.py   — operatory przestrzeni Hilberta
│   └── validation.py — walidacja macierzy gęstości
├── dataset/         — generator zbiorów danych (gałąź quantum_statesDB)
└── ml/              — moduł uczenia maszynowego (planowany)
```

## 4. Wymagania

```
contourpy==1.3.3
cycler==0.12.1
fonttools==4.62.1
kiwisolver==1.5.0
matplotlib==3.10.8
numpy==2.4.4
packaging==26.0
pillow==12.1.1
pyparsing==3.3.2
python-dateutil==2.9.0.post0
qutip==5.2.3
scipy==1.17.1
six==1.17.0
```

## 5. Instalacja

```pip install -r requirements.txt```

## 6. Przykłady użycia

### Python API

```
from src.physics.state.fock import FockState
from src.physics.measurement.wigner import WignerMeasurement

state = FockState(n=3, cutoff=32)
wm = WignerMeasurement(x_max=5, resolution=100)

W = wm.measure(state.density_matrix())
```

### CLI

```python generate.py --help```

## 7. Dostępne stany

- Stan dwumianowy
- Stan kota Schrödingera
- Stan koherentny
- Stan Focka
- Stan GKP
- Stan termiczny
- Stan próżni

## 8. Autorzy

- Karol Bartkiewicz
- Franciszek Grzywa
- Sebastian Dolata
- Mieszko Stryjski
- Julian Szamotuła
- Piotr Stećków

## 9. Wybrane publikacje opiekuna

1. P. Tulewicz, K. Bartkiewicz, A. Miranowicz, F. Nori, *"Resource-Efficient Quantum Correlation Measurement: A Multicopy Neural Network Approach"*, arXiv:2411.05745 (2024)
2. K. Bartkiewicz, P. Tulewicz, J. Roik, K. Lemr, *"Synergic quantum generative machine learning"*, Scientific Reports **13**, 12880 (2023)
3. K. Bartkiewicz, C. Gneiting, A. Černoch, K. Jiráková, K. Lemr, F. Nori, *"Experimental kernel-based quantum machine learning in finite feature space"*, Scientific Reports **10**, 12356 (2020)
4. V. Trávníček, K. Bartkiewicz, A. Černoch, K. Lemr, *"Experimental measurement of the Hilbert-Schmidt distance between two-qubit states"*, Physical Review Letters **123**, 260501 (2019)
