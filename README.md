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

Proponowany projekt ma na celu implementację i walidację metod głębokiego uczenia dla optycznych stanów kwantowych, obejmujących stany Focka, koherentne, termiczne, stany kota Schrödingera, binomialne oraz Gottesmana-Kitaeva-Preskilla, ze szczególnym uwzględnieniem robustności metod w obecności szumu eksperymentalnego.

## 2. Cele projektu

**Cel główny**: Opracowanie, implementacja i walidacja metod głębokiego uczenia maszynowego do klasyfikacji i rekonstrukcji optycznych stanów kwantowych w kontekście osiągnięć zespołu prof. Bartkiewicza w dziedzinie kwantowego uczenia maszynowego.

**Cele szczegółowe**:

1. Implementacja klasyfikatora na bazie konwolucyjnej sieci neuronowej do identyfikacji siedmiu klas stanów kwantowych na podstawie funkcji Husimi Q lub Wignera
2. Rozwój architektury warunkowej generatywnej sieci przeciwstawnej (cGAN) dla tomografii z niestandardowymi warstwami kwantowymi (macierz gęstości, wartości oczekiwane)
3. Analiza porównawcza z klasycznymi metodami estymacji oraz podejściem wielokopijnym zespołu prof. Bartkiewicza
4. Badanie odporności na różne typy szumu eksperymentalnego
5. Interpretacja decyzji sieci metodą Grad-CAM w celu optymalizacji strategii pomiarowych

## 3. Zadania naukowe i harmonogram (36 tygodni, 4 osoby)

### Faza I: Infrastruktura i fundamenty teoretyczne (Tygodnie 1–8)

Osoby zaangażowane: 4 (wszyscy członkowie zespołu)

- Studia literaturowe: analiza prac zespołu prof. Bartkiewicza i Ahmed et al.
- Konfiguracja środowiska (Python, TensorFlow/PyTorch, QuTiP)
- Implementacja symulatorów stanów kwantowych
- Generacja pierwszych zbiorów danych uczących (>10 000 stanów)
- Implementacja modeli szumu: strata fotonów, szum gaussowski, mieszanie stanów

**Rezultat**: Działające środowisko symulacyjne, wstępna baza danych, raport z analizy literatury.

### Faza II: Klasyfikacja stanów kwantowych (Tygodnie 9–18)

Podział zadań:
- **Osoba 1–2**: Projektowanie i implementacja architektury CNN
- **Osoba 3**: Rozszerzenie bazy danych, augmentacja danych
- **Osoba 4**: Implementacja metryk ewaluacyjnych, wizualizacja wyników

Zadania szczegółowe:
- Projektowanie architektury: warstwy konwolucyjne, warstwy łączące, warstwy w pełni połączone
- Uczenie klasyfikatora z walidacją krzyżową
- Optymalizacja hiperparametrów
- Analiza wydajności: macierze pomyłek, krzywe ROC
- Badanie wpływu szumu na dokładność klasyfikacji
- Implementacja i zastosowanie Grad-CAM

**Rezultat**: Działający klasyfikator z pełną dokumentacją wydajności, analiza obszarów decyzyjnych sieci.

### Faza III: Rekonstrukcja stanów — warunkowa generatywna sieć przeciwstawna (Tygodnie 19–30)

Podział zadań:
- **Osoba 1–2**: Implementacja sieci generującej z warstwami kwantowymi
- **Osoba 3**: Implementacja sieci dyskryminującej
- **Osoba 4**: Porównanie z metodami klasycznymi (MLE, APG)

Zadania szczegółowe:
- Projektowanie niestandardowych warstw:
  - Warstwa macierzy gęstości (hermitowska, dodatnio określona, ślad jednostkowy)
  - Warstwa wartości oczekiwanych (reguła Borna)
- Uczenie antagonistyczne z różnymi funkcjami kosztu
- Implementacja i testowanie klasycznych metod rekonstrukcji
- Analiza zbieżności i wierności rekonstrukcji
- Badanie wydajności dla stanów czystych i mieszanych
- Testy robustności na szum

**Rezultat**: Kompletny system tomografii z analizą porównawczą pokazującą przyspieszenie rekonstrukcji.

### Faza IV: Analiza, walidacja i dokumentacja (Tygodnie 31–36)

Osoby zaangażowane: 4

- Kompleksowa analiza wyników
- Porównanie z podejściem wielokopijnym prof. Bartkiewicza
- Testy na symulowanych danych eksperymentalnych
- Przygotowanie publikacji naukowej
- Dokumentacja kodu i repozytorium
- Przygotowanie prezentacji i plakatu konferencyjnego

**Rezultat**: Gotowa publikacja, udokumentowany kod, prezentacja.

## 4. Powiązanie z badaniami opiekuna projektu

Projekt stanowi kontynuację badań dr hab. Karola Bartkiewicza, prof. UAM:

- **Kwantowe uczenie maszynowe**: nawiązanie do synergicznego kwantowego generatywnego uczenia (*Scientific Reports*, 2023)
- **Tomografia wspierana uczeniem**: rozwinięcie metod z 67% redukcją wymagań pomiarowych (Tulewicz, Bartkiewicz et al., 2024)
- **Kwantowe metody jądrowe**: związek z eksperymentalną realizacją w skończonej przestrzeni cech (*Scientific Reports*, 2020)
- **Detekcja splątania**: wykorzystanie doświadczeń z projektowania świadków splątania sieciami neuronowymi (*PRL*, 2019)

## 5. Spodziewane rezultaty

### Rezultaty naukowe
- Artykuł w czasopiśmie JCR (*Physical Review A*, *Quantum Science and Technology*, *Scientific Reports*)
- Referat na konferencji międzynarodowej (IEEE QCE, QIP) lub krajowej (KSKIK)
- Poster konferencyjny

### Rezultaty techniczne
- Publicznie dostępne repozytorium z implementacjami sieci neuronowych, narzędziami symulacyjnymi i dokumentacją
- Interaktywne narzędzia wizualizacyjne (funkcja Wignera, macierze gęstości, mapy aktywacji)

### Rezultaty edukacyjne
Uczestnicy nabędą kompetencje w zakresie implementacji algorytmów uczenia maszynowego, symulacji układów kwantowych i analizy danych pomiarowych.

## 6. Struktura repozytorium

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

## 7. Wybrane publikacje opiekuna

1. P. Tulewicz, K. Bartkiewicz, A. Miranowicz, F. Nori, *"Resource-Efficient Quantum Correlation Measurement: A Multicopy Neural Network Approach"*, arXiv:2411.05745 (2024)
2. K. Bartkiewicz, P. Tulewicz, J. Roik, K. Lemr, *"Synergic quantum generative machine learning"*, Scientific Reports **13**, 12880 (2023)
3. K. Bartkiewicz, C. Gneiting, A. Černoch, K. Jiráková, K. Lemr, F. Nori, *"Experimental kernel-based quantum machine learning in finite feature space"*, Scientific Reports **10**, 12356 (2020)
4. V. Trávníček, K. Bartkiewicz, A. Černoch, K. Lemr, *"Experimental measurement of the Hilbert-Schmidt distance between two-qubit states"*, Physical Review Letters **123**, 260501 (2019)
