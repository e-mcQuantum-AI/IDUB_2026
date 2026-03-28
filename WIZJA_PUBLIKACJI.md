# Wizja publikacji naukowej

## Dane bibliograficzne

**Roboczy tytuł**: *Deep learning for classification and reconstruction of quantum optical states from phase-space representations*

**Autorzy**: [Członkowie zespołu], K. Bartkiewicz

**Czasopismo docelowe** (w kolejności preferencji):
1. **Physical Review A** — prestiż, zgodność tematyczna (quantum information + ML), IF ~2.9
2. **Quantum Science and Technology** — rosnące znaczenie w społeczności, IF ~6.7
3. **Scientific Reports** — otwarty dostęp, szybki proces recenzyjny, IF ~4.6

**Typ artykułu**: Regular Article (nie Letter — potrzebujemy miejsca na analizę 7 stanów i porównanie metod)

---

## Teza i wkład naukowy

### Główna teza

Konwolucyjne sieci neuronowe oraz warunkowe generatywne sieci przeciwstawne z niestandardowymi warstwami wymuszającymi fizyczność macierzy gęstości umożliwiają jednoczesną klasyfikację i rekonstrukcję optycznych stanów kwantowych z funkcji Wignera i Husimi Q z dokładnością porównywalną lub przewyższającą metody klasyczne, przy znacząco krótszym czasie obliczeń.

### Elementy nowości (novelty claims)

1. **Systematyczne porównanie Wigner vs Husimi Q** jako danych wejściowych do sieci neuronowych dla 7 klas stanów optycznych — dotychczasowe prace (Ahmed et al. 2021) skupiały się na jednej reprezentacji.

2. **Niestandardowe warstwy kwantowe w cGAN** — warstwa macierzy gęstości (DensityMatrixLayer) wymuszająca hermitowskość, jednostkowy ślad i dodatnią półokreśloność bezpośrednio w architekturze sieci. Istniejące prace stosują ograniczenia jako regularyzację w funkcji kosztu (post-hoc), a nie jako wbudowane ograniczenie strukturalne.

3. **Analiza odporności na szum dla wszystkich 7 typów stanów optycznych** — wcześniejsze prace analizowały 3–4 stany. Uwzględnienie stanów GKP i dwumianowych (kody korekcji błędów) w kontekście ML jest nowe.

4. **Porównanie CNN/cGAN z MLE i APG** na jednolitym zbiorze danych obejmującym stany czyste i mieszane z różnymi typami szumu eksperymentalnego.

### Czego praca NIE obiecuje

- Nie jest to praca eksperymentalna — używamy danych symulowanych (QuTiP).
- Nie implementujemy podejścia wielokopijnego (to osobny kierunek badań grupy).
- Nie twierdzimy, że nasza metoda jest optymalna — pokazujemy, że jest praktyczna i szybka.

---

## Struktura artykułu

### I. Introduction (1.5 strony)

**Narracja**: Od ogólnego problemu tomografii kwantowej → przez ograniczenia metod klasycznych → do propozycji uczenia maszynowego.

Akapit 1 — *Kontekst*:
- Tomografia kwantowa (QST) jako fundamentalna procedura w informacji kwantowej
- Skalowanie: pełna tomografia wymaga O(d²) pomiarów, d = wymiar przestrzeni Hilberta
- Tradycyjne metody (MLE, APG) są kosztowne obliczeniowo

Akapit 2 — *Stan wiedzy*:
- Sieci neuronowe w tomografii: przegląd kluczowych prac
  - Ahmed et al. (2021) — CNN + GAN, 98% accuracy, F > 0.99 [główna inspiracja]
  - Torlai et al. (2018) — tomografia z sieciami RBM
  - Neugebauer et al. (2020) — ML dla stanów ciągłych
- Prace grupy Bartkiewicza:
  - Tulewicz et al. (2024) — podejście wielokopijne z NN, 67% redukcja pomiarów
  - Bartkiewicz et al. (2023) — synergiczne generatywne uczenie kwantowe
  - Trávníček et al. (2019) — pomiar odległości Hilberta-Schmidta

Akapit 3 — *Luka badawcza*:
- Brak systematycznego porównania reprezentacji fazowych (Wigner vs Husimi Q) jako danych wejściowych
- Istniejące architektury GAN nie wymuszają fizyczności macierzy gęstości strukturalnie
- Brak analizy obejmującej jednocześnie stany kodów korekcji błędów (GKP, dwumianowe) i stany klasyczne (koherentne, termiczne)

Akapit 4 — *Nasza propozycja*:
- CNN do klasyfikacji 7 stanów optycznych
- cGAN z warstwami wymuszającymi fizyczność do rekonstrukcji
- Systematyczna analiza odporności na 5 typów szumu
- Porównanie z MLE i APG

### II. Theoretical background (2 strony)

**II.A Stany optyczne** — definicje matematyczne 7 stanów:
- Próżnia: $\ket{0}$
- Fock: $\ket{n}$
- Koherentny: $\ket{\alpha} = e^{-|\alpha|^2/2} \sum_n \frac{\alpha^n}{\sqrt{n!}} \ket{n}$
- Kot Schrödingera: $\ket{\text{cat}} \propto \ket{\alpha} + \ket{-\alpha}$
- Termiczny: $\rho_\text{th} = \sum_n \frac{\bar{n}^n}{(\bar{n}+1)^{n+1}} \ket{n}\bra{n}$
- Dwumianowy: $\ket{\psi_\text{bin}} = \sum_{n=0}^{N} \sqrt{\binom{N}{n}} p^{n/2} (1-p)^{(N-n)/2} \ket{n}$ [ref: Michael et al. 2016]
- GKP: $\ket{\text{GKP}} \propto \sum_s e^{-2\pi s^2/\delta^2} \hat{D}(s\sqrt{2\pi}) \ket{0}$ [ref: Gottesman et al. 2001]

**II.B Reprezentacje w przestrzeni fazowej**:
- Funkcja Wignera: $W(q,p) = \frac{1}{\pi\hbar} \int \bra{q+y}\rho\ket{q-y} e^{2ipy/\hbar} dy$
- Funkcja Husimi Q: $Q(\alpha) = \frac{1}{\pi} \bra{\alpha}\rho\ket{\alpha}$
- Porównanie: Wigner może być ujemna (nieklasyczność), Husimi Q zawsze ≥ 0
- **Tabela**: właściwości obu reprezentacji (rozdzielczość informacyjna, szum, koszt obliczeniowy)

**II.C Modele szumu**:
- Straty fotonowe: równanie Lindblada z $\hat{c} = \sqrt{\gamma}\hat{a}$
- Defazowanie: $\hat{c} = \sqrt{\gamma}\hat{n}$
- Depolaryzacja: $\rho \to (1-p)\rho + p \mathbf{I}/d$
- Szum gaussowski: addytywny na kwadraturach
- Szum termiczny: mieszanie ze stanem termicznym

**II.D Metody klasyczne** (krótko):
- MLE — estymacja największej wiarygodności
- APG — przyspieszona metoda rzutowanego gradientu

### III. Methods (3 strony)

**III.A Generowanie danych** — opis procedury:
- 7 klas × 2000 próbek na klasę = 14 000 stanów
- 5 poziomów szumu: γ ∈ {0, 0.05, 0.1, 0.2, 0.5}
- 5 typów szumu × 5 poziomów = 25 konfiguracji szumu + czyste dane
- Rozdzielczość: 64×64 pikseli
- Podział: 70% train / 15% val / 15% test (stratyfikowany)
- **Tabela**: zakresy parametrów dla każdego stanu (α, n, n_th, N, p, δ)

**III.B Architektura CNN** (z diagramem):
- Wejście: (1, 64, 64)
- 4 bloki: Conv2d(32/64/128/256) → BatchNorm → ReLU → MaxPool(2×2)
- Global Average Pooling
- FC(256 → 128) → Dropout(0.3) → FC(128 → 7) → Softmax
- Cross-entropy loss, Adam optimizer, lr = 1e-3 z CosineAnnealing
- **Rysunek 1**: Schemat architektury CNN

**III.C Architektura cGAN** (z diagramem):
- **Generator** (Encoder-Decoder, U-Net):
  - Wejście: zaszumiona funkcja Wignera (1, 64, 64) + one-hot etykieta klasy (7)
  - Encoder: Conv2d downsampling (64→128→256→512)
  - Bottleneck: DensityMatrixLayer — wymuszenie fizyczności
  - Decoder: ConvTranspose2d upsampling z skip connections
  - Wyjście: zrekonstruowany obraz Wignera (1, 64, 64)
- **Dyskryminator** (PatchGAN):
  - Wejście: obraz Wignera + etykieta
  - 4 bloki Conv2d → LeakyReLU
  - Wyjście: mapa 8×8 (prawdziwy/fałszywy per patch)
- **Funkcja kosztu**:
  $$\mathcal{L} = \mathcal{L}_\text{adv} + \lambda_1 \mathcal{L}_\text{L1} + \lambda_F \mathcal{L}_\text{fidelity}$$
  - $\mathcal{L}_\text{adv}$: adversarial loss (LSGAN)
  - $\mathcal{L}_\text{L1}$: rekonstrukcja pikselowa
  - $\mathcal{L}_\text{fidelity}$: $1 - F(\rho_\text{pred}, \rho_\text{true})$
- **Rysunek 2**: Schemat architektury cGAN z zaznaczeniem warstw kwantowych

**III.D Warstwa macierzy gęstości** (kluczowa innowacja):
- Parametryzacja Cholesky'ego: $\rho = T^\dagger T / \text{Tr}(T^\dagger T)$
- Gwarancje: hermitowskość ($\rho = \rho^\dagger$), jednostkowy ślad, dodatnia półokreśloność
- Gradient przechodzi przez dekompozycję — warstwa jest w pełni różniczkowalna
- **Rysunek 3**: Schemat działania DensityMatrixLayer

**III.E Metryki**:
- Klasyfikacja: accuracy, precision, recall, F1-score, macierz pomyłek
- Rekonstrukcja: wierność $F(\rho_1, \rho_2) = \left(\text{Tr}\sqrt{\sqrt{\rho_1}\rho_2\sqrt{\rho_1}}\right)^2$, trace distance, odległość Hilberta-Schmidta

### IV. Results (4 strony)

**IV.A Klasyfikacja — dane czyste**:
- **Tabela 1**: Accuracy per klasa (7 stanów), porównanie Wigner vs Husimi Q
- Oczekiwany wynik: accuracy > 95% (ref: Ahmed et al. osiągnęli 98%)
- **Rysunek 4**: Macierz pomyłek (confusion matrix) — dwa panele: Wigner | Husimi Q
- Analiza: które pary stanów są najczęściej mylone (prawdopodobnie koherentny↔próżnia przy małym α)

**IV.B Klasyfikacja — dane zaszumione**:
- **Rysunek 5**: Accuracy vs γ dla każdego typu szumu (5 krzywych na wykresie)
- **Rysunek 6**: Heatmapa accuracy(typ stanu, typ szumu) przy γ = 0.2
- Analiza: które stany są najbardziej wrażliwe na szum (hipoteza: GKP i kot degradują szybciej niż koherentny i termiczny)
- Porównanie: trening na czystych vs trening na zaszumionych danych (transfer learning argument)
- **Tabela 2**: Accuracy dla kluczowych konfiguracji (czyste, γ=0.1, γ=0.5)

**IV.C Rekonstrukcja cGAN**:
- **Tabela 3**: Wierność rekonstrukcji per stan, per poziom szumu
- Oczekiwany wynik: F > 0.95 dla stanów czystych z umiarkowanym szumem
- **Rysunek 7**: Przykłady rekonstrukcji — 3 kolumny: oryginał | zaszumiony | zrekonstruowany — dla wybranych stanów
- **Rysunek 8**: Wierność vs γ (7 krzywych, jedna per stan)

**IV.D Porównanie z metodami klasycznymi**:
- **Tabela 4**: CNN/cGAN vs MLE vs APG — wierność i czas obliczeniowy
- **Rysunek 9**: Scatter plot: wierność vs czas (log scale) — każdy punkt = metoda × stan
- Oczekiwany wynik: cGAN osiąga porównywalną wierność do MLE przy 10–100× krótszym czasie
- Analiza: w jakich warunkach metody ML wygrywają (duży szum, wiele stanów), a w jakich przegrywają (bardzo czyste dane, pojedynczy stan)

**IV.E Wpływ architektury**:
- Porównanie: prosty CNN vs ResNet-18 fine-tuned
- Porównanie: rozdzielczość 32×32 vs 64×64 vs 128×128
- Ablation study: cGAN z DensityMatrixLayer vs bez (regularyzacja w loss vs warstwa strukturalna)

### V. Discussion (1.5 strony)

Akapit 1 — *Podsumowanie głównych wyników*:
- CNN skutecznie klasyfikuje 7 stanów optycznych z obu reprezentacji fazowych
- cGAN z warstwami kwantowymi rekonstruuje stany z wysoką wiernością
- Przewaga szybkości nad metodami klasycznymi

Akapit 2 — *Wigner vs Husimi Q*:
- Która reprezentacja jest lepsza i dlaczego (hipoteza: Wigner lepsza dla stanów nieklasycznych dzięki ujemnym wartościom, Husimi lepsza dla stanów mieszanych dzięki gładkości)
- Implikacje praktyczne: wybór pomiaru w eksperymencie

Akapit 3 — *Warstwy kwantowe*:
- Ablation study potwierdza, że wbudowane ograniczenie fizyczności jest lepsze niż regularyzacja
- Konsekwencja: wygenerowane macierze gęstości są zawsze fizyczne — nie wymaga post-processingu

Akapit 4 — *Odporność na szum*:
- Które stany są najtrudniejsze i dlaczego (związek z nieklasycznością, energią)
- Praktyczne implikacje: minimalny stosunek sygnał/szum dla różnych stanów

Akapit 5 — *Ograniczenia i perspektywy*:
- Dane symulowane — potrzebna walidacja na danych eksperymentalnych
- Skalowanie: jak metoda zachowuje się dla większych przestrzeni Hilberta (cutoff > 40)
- Możliwe rozszerzenia: stany wielomodowe, tomografia procesów kwantowych
- Związek z podejściem wielokopijnym (Tulewicz, Bartkiewicz et al. 2024) — potencjalna synergia

### VI. Conclusions (0.5 strony)

- Streszczenie: co zrobiliśmy, co pokazaliśmy, co to oznacza
- Kluczowe liczby: accuracy, wierność, przyspieszenie
- Kod publicznie dostępny: link do repozytorium

### Appendices (opcjonalnie)

- A: Pełne tabele wyników (wszystkie kombinacje stanów × szumów × γ)
- B: Szczegóły architektur sieci (liczba parametrów, hiperparametry)
- C: Dodatkowe wykresy

---

## Rysunki i tabele — plan

### Rysunki (9–10)

| # | Opis | Typ | Etap |
|---|------|-----|------|
| 1 | Architektura CNN (schemat blokowy) | Diagram | 3 |
| 2 | Architektura cGAN z warstwami kwantowymi | Diagram | 4 |
| 3 | DensityMatrixLayer — schemat działania | Diagram | 4 |
| 4 | Macierz pomyłek: Wigner vs Husimi Q | Heatmapa 2-panelowa | 7 |
| 5 | Accuracy vs γ per typ szumu | Wykresy liniowe | 7 |
| 6 | Heatmapa accuracy (stan × szum) | Heatmapa | 7 |
| 7 | Przykłady rekonstrukcji: oryginał / zaszumiony / zrekonstruowany | Grid obrazów | 7 |
| 8 | Wierność rekonstrukcji vs γ per stan | Wykresy liniowe | 7 |
| 9 | Wierność vs czas: CNN/cGAN vs MLE vs APG | Scatter plot | 7 |

### Tabele (4–5)

| # | Opis | Etap |
|---|------|------|
| 1 | Accuracy klasyfikacji per stan (czyste dane, Wigner vs Husimi) | 7 |
| 2 | Accuracy w kluczowych konfiguracjach szumu | 7 |
| 3 | Wierność rekonstrukcji per stan × szum | 7 |
| 4 | Porównanie metod: wierność, czas, skalowanie | 7 |

---

## Kluczowe referencje

Artykuł powinien cytować następujące prace (minimum):

### Bezpośrednie inspiracje
- Ahmed et al., *"Classification and reconstruction of optical quantum states with deep neural networks"*, Physical Review Research **3**, 033278 (2021)
- Tulewicz, Bartkiewicz, Miranowicz, Nori, *"Resource-Efficient Quantum Correlation Measurement"*, arXiv:2411.05745 (2024)
- Bartkiewicz et al., *"Synergic quantum generative machine learning"*, Scientific Reports **13**, 12880 (2023)

### Tomografia kwantowa — metody klasyczne
- Hradil, *"Quantum-state estimation"*, PRA **55**, R1561 (1997) — MLE
- Gross et al., *"Quantum state tomography via compressed sensing"*, PRL **105**, 150401 (2010)

### ML w fizyce kwantowej
- Torlai et al., *"Neural-network quantum state tomography"*, Nature Physics **14**, 447 (2018)
- Carleo, Troyer, *"Solving the quantum many-body problem with artificial neural networks"*, Science **355**, 602 (2017)

### Stany optyczne
- Gottesman, Kitaev, Preskill, *"Encoding a qubit in an oscillator"*, PRA **64**, 012310 (2001) — GKP
- Michael et al., *"New class of quantum error-correcting codes for a bosonic mode"*, PRX **6**, 031006 (2016) — stan dwumianowy

### Narzędzia
- Johansson et al., *"QuTiP 2"*, Computer Physics Communications **184**, 1234 (2013)

### Prace grupy Bartkiewicza (kontekst)
- Trávníček, Bartkiewicz et al., *"Experimental measurement of the Hilbert-Schmidt distance"*, PRL **123**, 260501 (2019)
- Bartkiewicz et al., *"Experimental kernel-based quantum machine learning"*, Scientific Reports **10**, 12356 (2020)

---

## Cele liczbowe (benchmarki do osiągnięcia)

Na podstawie Ahmed et al. (2021) i specyfiki naszego zbioru stanów:

| Metryka | Cel minimalny | Cel ambitny | Ahmed et al. |
|---------|---------------|-------------|--------------|
| Accuracy klasyfikacji (czyste) | > 95% | > 98% | 98.3% |
| Accuracy klasyfikacji (γ=0.1) | > 85% | > 92% | nie badano |
| Wierność rekonstrukcji (czyste) | > 0.95 | > 0.99 | 0.993 |
| Wierność rekonstrukcji (γ=0.1) | > 0.90 | > 0.95 | nie badano |
| Przyspieszenie vs MLE | > 10× | > 100× | ~100× |

---

## Harmonogram pisania

| Tydzień projektu | Sekcja | Kto |
|------------------|--------|-----|
| 22–24 | Methods (III) — po zakończeniu implementacji | Osoba 1 + 2 |
| 24–26 | Results (IV.A–B) — klasyfikacja | Osoba 1 |
| 26–28 | Results (IV.C–E) — rekonstrukcja i porównanie | Osoba 2 |
| 28–30 | Theoretical background (II) | Osoba 3 |
| 30–32 | Introduction (I), Discussion (V) | Osoba 4 + opiekun |
| 32–34 | Rysunki, tabele, formatowanie | Wszyscy |
| 34–36 | Rewizja, korekta, submission | Opiekun + wszyscy |

---

## Ryzyka i plany awaryjne

| Ryzyko | Prawdopodobieństwo | Plan awaryjny |
|--------|-------------------|----------------|
| Accuracy CNN < 90% | Niskie | Zwiększyć zbiór danych, próbować ResNet/EfficientNet |
| cGAN nie zbiega się | Średnie | Uprościć do autoencodera (bez adversarial), użyć WGAN-GP |
| Wierność cGAN < 0.9 | Średnie | Skupić się na stanach czystych, zwiększyć wagę fidelity loss |
| MLE za wolne dla porównania | Niskie | Ograniczyć do mniejszych cutoff (N=20), podać extrapolację |
| Recenzent żąda danych eksperymentalnych | Wysokie | Dodać sekcję o symulacji szumu realistycznego, zapowiedzieć walidację eksperymentalną |
| Recenzent kwestionuje nowość | Średnie | Podkreślić: (a) DensityMatrixLayer, (b) 7 stanów, (c) Wigner vs Husimi |
