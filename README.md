# Segmentacja Guza Mózgu — BraTS2020

> Projekt z przedmiotu Sieci Neuronowe  
> Autorzy: [Imię 1], [Imię 2]  
> Repozytorium: https://github.com/Kapponohacco/Neural-networks-brain-tumor-segmentation  
> Prezentacja wyników: [link do prezentacji]

---

## Spis treści

1. [Opis projektu](#opis-projektu)
2. [Zbiór danych](#zbiór-danych)
3. [Inspekcja danych](#inspekcja-danych)
4. [Przebieg projektu](#przebieg-projektu)
   - [Etap 1 — Pierwsze ładowanie danych i lazy loader](#etap-1--pierwsze-ładowanie-danych-i-lazy-loader)
   - [Etap 2 — Eksperymenty z funkcją straty](#etap-2--eksperymenty-z-funkcją-straty)
   - [Etap 3 — Preprocessing i cache'owanie](#etap-3--preprocessing-i-cacheowanie)
   - [Etap 4 — Trening w chmurze i ulepszone przycinanie](#etap-4--trening-w-chmurze-i-ulepszone-przycinanie)
   - [Etap 5 — Finalne modele i dostrajanie wag klas](#etap-5--finalne-modele-i-dostrajanie-wag-klas)
5. [Architektura modeli](#architektura-modeli)
6. [Wyniki](#wyniki)

---

## Opis projektu

Celem projektu jest segmentacja guzów mózgu na skanach MRI z wykorzystaniem sieci neuronowych typu U-Net. Model klasyfikuje każdy piksel skanu do jednej z czterech klas:

| Klasa | Etykieta | Opis |
|---|---|---|
| Tło | 0 | Obszar poza mózgiem |
| Martwica / NCR | 1 | Martwicze i nie wzmacniające się jądro guza |
| Obrzęk | 2 | Okołoguzowy obrzęk tkanki |
| Guz wzmacniający | 3 | Aktywnie wzmacniający się guz (oryginalna etykieta 4 → przemapowana na 3) |

---

## Zbiór danych

Korzystamy z publicznego zbioru **BraTS2020** (Brain Tumor Segmentation Challenge 2020).

- **369 skanów** mózgów w zestawie treningowym
- Każdy skan: **155 przekrojów (slice'ów)** o rozdzielczości **240×240 pikseli**
- **4 modalności MRI** + maska segmentacji:
  - **T1** — szczegóły anatomiczne tkanki mózgowej
  - **T1ce** — T1 z kontrastem; uwydatnia aktywne obszary guza
  - **T2** — uwydatnia płyny i obrzęki
  - **FLAIR** — jak T2, ale z tłumieniem sygnału płynu mózgowo-rdzeniowego
- Łączny rozmiar zbioru: **~40 GB** w formacie NIfTI (`.nii`)

---

## Inspekcja danych

Przed przystąpieniem do treningu przeprowadziliśmy inspekcję zbioru danych w celu zrozumienia jego struktury i właściwości. Najważniejsze obserwacje:

**Format NIfTI** — dane są zapisane w formacie `.nii`, który przechowuje trójwymiarowe wolumeny wraz z metadanymi przestrzennymi (orientacja, rozmiar voksela). Do ładowania używamy biblioteki `nibabel`.

**Brak brakujących slice'ów** — wszystkie 369 skanów zawiera dokładnie 155 przekrojów, więc zbiór jest kompletny.

**Rozkład etykiet** — sprawdziliśmy liczebność każdej klasy w maskach segmentacji. Dominującą etykietą jest tło (klasa 0), które stanowi zdecydowaną większość wszystkich vokseli (~3,2 miliarda). Etykieta 3 (brakujące voksele) nie wystąpiła w żadnym skanie, co pozwoliło nam bezpiecznie przemapować etykietę 4 (guz wzmacniający) na etykietę 3.

**Puste slice'y** — znaczna część każdego skanu (slice'y na początku i końcu wolumenu) to obszary poza mózgiem, które nie wnoszą żadnej informacji do treningu.

**Bounding box mózgu** — po przejrzeniu wszystkich skanów wyznaczyliśmy minimalny prostokąt obejmujący piksele mózgu we wszystkich wolumenach:

```
x: [41, 195]  →  szerokość:  155 px
y: [29, 222]  →  wysokość:  194 px
```

---

## Przebieg projektu

### Etap 1 — Pierwsze ładowanie danych i lazy loader

Pierwszym napotkanym problemem były wymagania pamięciowe związane z przechowywaniem danych MRI. Początkowe eksperymenty zakładały wczytywanie większej liczby wolumenów jednocześnie, co szybko prowadziło do nadmiernego zużycia pamięci i problemów ze stabilnością procesu treningowego.

Aby ograniczyć ilość danych znajdujących się jednocześnie w pamięci, zaimplementowaliśmy lazy loadera, który wczytywał tylko aktualnie potrzebne fragmenty danych. Pozwoliło to kontynuować eksperymenty bez konieczności przechowywania wielu pełnych wolumenów MRI jednocześnie.

Rozwiązanie to ujawniło jednak kolejny problem — przygotowanie danych zaczęło zajmować więcej czasu niż sam trening modelu. Wielokrotne odczytywanie plików .nii sprawiało, że GPU często oczekiwało na przygotowanie kolejnych batchy przez CPU oraz odczyt danych z dysku.

---

### Etap 2 — Eksperymenty z funkcją straty

Nie chcąc utknąć na problemach z ładowaniem danych, równolegle prowadziliśmy eksperymenty z **funkcjami straty i wagami klas**. Dominacja pikseli tła w danych sprawiała, że model bez specjalnej obsługi tej nierównowagi uczył się ignorować rzadkie klasy guza.

Przetestowaliśmy następujące podejścia:

- **CrossEntropyLoss** (bez wag) — baseline, model zbyt mocno skupiał się na tle
- **Ważony CrossEntropyLoss** — przypisanie wyższych wag rzadkim klasom guza; wagi `[0.1, 2.0, 2.0, 2.0]` dały dobre wyniki przy niskim narzucie czasowym
- **DiceLoss (MONAI)** — poprawny dla segmentacji medycznej, jednak zauważalnie wydłużał czas treningu względem CrossEntropyLoss
- **DiceCELoss (MONAI)** — kombinacja Dice i CrossEntropy z opcjonalnymi wagami i parametrem `gamma` dla trudnych przykładów; wyniki podobne do ważonego CE, ale wolniejszy trening

Ostatecznie wybraliśmy **ważony `nn.CrossEntropyLoss`** z wagami `[0.1, 2.0, 2.0, 2.0]` jako optymalny kompromis między jakością wyników a szybkością treningu.

Testy te były prowadzone na **maksymalnie 120 mózgach i 35 epokach** z jedną modalnością i podstawową architekturą U-Net bez normalizacji. Jeden taki trening zajmował około 350 minut, przez co liczba eksperymentów była ograniczona.

---

### Etap 3 — Preprocessing i cache'owanie

Aby wyeliminować problem z CPU z Etapu 1, wdrożyliśmy **preprocessing offline z cache'owaniem do plików `.pt`**.

Pipeline preprocessingu:

1. Wczytanie pliku `.nii` przez `nibabel`
2. **Normalizacja** per-modalność
3. **Przeskalowanie** slice'ów do rozmiaru **128×128**
4. Zapis każdego mózgu do pliku `.pt` (`torch.save`)

Dzięki temu podczas treningu możliwe było **wczytanie wszystkich mózgów do RAM-u** jednorazowo, co całkowicie wyeliminowało oczekiwanie na dane. Przy tej okazji dodaliśmy też **drugą modalność** — uznaliśmy, że `flair` i `t1ce` najlepiej uwydatniają różne typy zmian i komórek nowotworowych.

Dodatkowo zaimplementowaliśmy w `CustomDataset` logikę **pomijania 80% slice'ów z samym tłem** — każdy slice bez żadnego piksela guza był zachowywany z prawdopodobieństwem 20%, co przyspieszyło trening i poprawiło balans danych.

Trening: ~120 mózgów, 40 epok, batch = 1 mózg, czas ~315 minut.

---

### Etap 4 — Trening w chmurze i ulepszone przycinanie

Kolejnym planowanym krokiem było dodanie **Batch Normalization** do modelu oraz zwiększenie batch size z 1 do 3–4 mózgów. Testy diagnostyczne na 3 epokach wykazały jednak, że taki trening lokalnie zajmowałby **powyżej 12 godzin**.

Zdecydowaliśmy się przenieść trening do chmury:

- Cały projekt spakowany do **obrazu Dockera**
- Trening uruchomiony na **wynajętym GPU** przez platformę **RunPod**
- Zbiór danych dostarczany przez obraz Dockera

Jednocześnie ulepszyliśmy preprocessing, uwzględniając wyznaczony wcześniej bounding box:

1. **Przycięcie** każdego slice'a do `x: [41, 196], y: [29, 223]` — eliminacja pustego tła wokół mózgu
2. **Padding** do kwadratowego wymiaru — wyrównanie szerokości i wysokości
3. **Skalowanie** do **128×128**

Finalne parametry treningu w chmurze: **369 mózgów**, **40 epok**, **batch size = 3**.

---

### Etap 5 — Finalne modele i dostrajanie wag klas

W finalnych eksperymentach porównaliśmy trzy architektury:

**UNet** — bazowa architektura bez normalizacji (punkt odniesienia z wcześniejszych etapów).

**UNetNorm** — U-Net z warstwami **Batch Normalization** po każdej konwolucji. Stabilizuje trening i przyspiesza zbieżność.

**UNetResNet** — U-Net z enkoderem zastąpionym przez **ResNet-34** (pretrenowany na ImageNet). Trening realizowany w **trzech etapach stopniowego odmrażania warstw** enkodera:
- **Etap 1:** trenowany tylko dekoder (enkoder zamrożony)
- **Etap 2:** odblokowane `layer3` i `layer4` z LR × 0.1
- **Etap 3:** odblokowany pełny enkoder; `layer0–2` z LR × 0.01, `layer3–4` z LR × 0.1, dekoder z bazowym LR

Dodatkowo przetestowaliśmy **dostrajanie wag klas** bazujące na rzeczywistej częstotliwości ich występowania w zbiorze treningowym (wagi obliczane jako pierwiastek kwadratowy z odwrotności częstości), zamiast ręcznie dobranych wag `[0.1, 2.0, 2.0, 2.0]`.

---

## Architektura modeli

Wszystkie modele jako wejście przyjmują **2 kanały** (modalności `flair` + `t1ce`) i klasyfikują każdy piksel do jednej z **4 klas**.

### UNet / UNetNorm

Klasyczna architektura U-Net z 4 poziomami downsamplingu i symetrycznym dekoderem z skip connections. `UNetNorm` dodaje `BatchNorm2d` po każdej warstwie konwolucyjnej.

```
Wejście:  (B, 2, 128, 128)
Enkoder:  2 → 32 → 64 → 128 → 256 → 512  (z MaxPool2d)
Dekoder:  512 → 256 → 128 → 64 → 32
Wyjście:  (B, 4, 128, 128)  (logity dla 4 klas)
```

### UNetResNet

Enkoder oparty na **ResNet-34** (pretrenowanym), dekoder taki sam jak w UNetNorm.

```
Wejście:        (B, 2, 128, 128)
layer0_conv:    2 → 64   (Conv7×7, BN, ReLU)
layer0_pool:    MaxPool → 32×32
layer1–4:       ResNet-34 (64→64→128→256→512)
Dekoder:        512 → 256 → 128 → 64 → 32 → 32
Wyjście:        (B, 4, 128, 128)
```

---

## Wyniki

Metryka: **Dice score** per klasa, wyznaczony na zbiorze walidacyjnym (10% danych, ~37 mózgów).

| Model | Martwica | Guz wzmacniający | Obrzęk | Tło |
|---|---|---|---|---|
| UNet (baseline) | 0.566 | 0.760 | 0.582 | 0.989 |
| UNet (Tuned Weights) | 0.538 | 0.752 | 0.652 | 0.992 |
| UNetNorm | 0.604 | 0.817 | 0.674 | 0.993 |
| **UNetNorm (Tuned Weights)** | **0.625** | **0.818** | **0.721** | **0.995** |
| UNetResNet (Tuned Weights) | 0.595 | 0.791 | 0.726 | 0.995 |

Najlepszym modelem okazał się **UNetNorm z dostrajanymi wagami klas**, który osiągnął najwyższe wyniki dla martwicy oraz guza wzmacniającego, zachowując jednocześnie bardzo dobre rezultaty dla obrzęku.

Model **UNetResNet** osiągnął porównywalną jakość segmentacji, uzyskując najlepszy wynik dla klasy obrzęku (Dice = 0.726). Wynik ten sugeruje, że wykorzystanie pretrenowanego enkodera ResNet-34 jest obiecującym kierunkiem, który mógłby przynieść dalsze korzyści przy dłuższym treningu lub większej liczbie eksperymentów z dostrajaniem warstw.

Obserwacje:
- Dodanie Batch Normalization (UNet → UNetNorm) dało znaczącą poprawę we wszystkich klasach, szczególnie dla guza wzmacniającego (+5.7 pp) i obrzęku (+9.2 pp)
- Dostrajanie wag klas na podstawie częstotliwości poprawiło segmentację obrzęku we wszystkich modelach, kosztem nieznacznego spadku na martwicy
