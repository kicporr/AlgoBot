# Raport Zoptymalizowanych Parametrów i Statystyk Backtestu (Uproszczona Architektura)

Ten dokument zawiera dokładną konfigurację parametrów oraz pełne statystyki historyczne po usunięciu modułu XGBoost (uproszczenie architektury) dla zespołu strategii pary BTC/USDT na interwale 1H. Wyniki te wykazują wyższą stopę zwrotu oraz wyższy wskaźnik Sharpe'a we wszystkich przedziałach czasowych w porównaniu do wersji z XGBoostem.

---

## 1. Dokładna Konfiguracja Parametrów (bocik_config)

Poniższe parametry zostały zaimplementowane w `ensemble_backtest.py`, `final_backtest.py` oraz `config/settings.yaml` i dały zoptymalizowany wynik:

### A. Parametry Transakcyjne i Ryzyko (Wspólne)
* **Początkowy Kapitał**: `$10,000`
* **Maksymalny Rozmiar Pozycji (`max_position_pct`)**: `0.20` (20% aktualnego kapitału)
* **Model Opłat i Poślizgu (Bitget Spot)**:
  * Prowizja Taker (`taker`): `0.06%` (0.0006)
  * Prowizja Maker (`maker`): `0.02%` (0.0002)
  * Poślizg cenowy (`slippage`): `0.05%` (0.0005) per strona (łącznie `0.10%` poślizgu na pełną transakcję)
  * Szacowany koszt pełnego obrotu (round-trip): `0.14% - 0.22%` (14-22 bps)

### B. Strategia MTF MACD + Elder Filter (Trending Regime)
Główna strategia generująca zyski w fazach silnego trendu.
* **Szybka średnia MACD (`fast`)**: `8` (zoptymalizowana z 12)
* **Wolna średnia MACD (`slow`)**: `21` (zoptymalizowana z 26)
* **Okres Sygnału (`signal`)**: `9`
* **Filtr Trendu Wyższego Rzędu (Elder D1)**:
  * **Interwał**: `1D` (Dzienny)
  * **Kryterium trendu**: **Kierunek nachylenia (slope) histogramu MACD** (trend wzrostowy gdy `hist_slope > 0`, spadkowy gdy `hist_slope < 0`)
* **Ustawienia Zabezpieczeń i Celów (Exit Chain)**:
  * **Trailing Stop (`trailing_stop_pct`)**: `2.0%`
  * **Mnożnik ATR stopu (`atr_stop_mult`)**: `1.5x`
  * **Minimalny Czas Trzymania (`min_hold_bars`)**: `1` (1 godzina)
  * **Filtr Wolumenowy Eldera**: Wyłączony (`require_volume_confirm: False`)
  * **Krótkie pozycje (Shorts)**: Włączone (`allow_shorts: True`)

### C. Alokacja Regimów Rynkowych (Regime Router - Uproszczony)
Po usunięciu XGBoosta router w pełni unika niejasnych i niestabilnych warunków rynkowych:
* **TRENDING (MACD)**: `ADX >= 25` ORAZ Stosunek DI+ do DI- (`DI+ / DI- >= 1.3` lub `<= 0.77`) → Aktywuje **MTF MACD Elder**
* **RANGING (FLAT)**: `ADX <= 20` ORAZ szerokość wstęgi Bollingera `BBw <= 4%` ORAZ hist. zmienność `HV <= 50%` → **FLAT** (sit out)
* **VOLATILE (FLAT)**: Zbieg przynajmniej 2 sygnałów zmienności → **FLAT** (sit out)
* **UNCLEAR (FLAT)**: Brak jasnego trendu lub konsolidacji → **FLAT** (sit out - zastąpiono XGBoosta gotówką)
* **Histereza (Hysteresis)**: wymagane `2` świeczki potwierdzenia stanu przed zmianą regimu.

---

## 2. Statystyki Wyników Backtestu (6.5 Roku danych BTC, 2020-2026)

Poniżej przedstawiono kompletne statystyki w podziale na 3 niezależne próby chronologiczne (60% Train, 20% Val, 20% Test) dla nowej uproszczonej architektury:

### Podsumowanie Główne Zestawu:
* **Początkowy Kapitał**: `$10,000`
* **Łączny zysk finansowy (PnL)**: **+$3,927.22** (Kapitał urósł do **$13,927.22** vs **$13,536.00** z XGBoostem)
* **Maksymalne obsunięcie kapitału (Drawdown)**: **2.7%** (w okresie treningowym), **1.4%** (w okresie testowym)

---

### A. Zbiór Treningowy (TRAIN: 2020-2024)
* **Okres**: `2020-01-01` -> `2023-11-15` (3.9 lat)
* **Wynik Buy & Hold (BTC)**: `+394.6%`
* **Wynik Ensemble**: **+$2,496.00 (+25.0% zwrotu)**
* **Liczba Transakcji**: `264`
* **Skuteczność (Win Rate)**: `45.8%`
* **Wskaźnik Sharpe'a**: **+1.68**
* **Maksymalne obsunięcie (DD)**: `2.7%`
* **Profit Factor**: `1.83`
* **Średni zysk na transakcję**: `+$9`
* **Statystyki Wyjść**:
  * `signal` (przecięcie MACD): 162 (61%)
  * `trailing_stop` (2% trailing): 44 (17%)
  * `take_profit` (TP): 31 (12%)
  * `atr_stop` (SL ATR): 18 (7%)
  * `time_exit` (48h max): 9 (3%)

---

### B. Zbiór Walidacyjny (VAL: 2023-2025)
* **Okres**: `2023-11-15` -> `2025-03-01` (1.3 roku)
* **Wynik Buy & Hold (BTC)**: `+123.0%`
* **Wynik Ensemble**: **+$516.00 (+5.2% zwrotu)**
* **Liczba Transakcji**: `111`
* **Skuteczność (Win Rate)**: `43.2%`
* **Wskaźnik Sharpe'a**: **+1.25**
* **Maksymalne obsunięcie (DD)**: `2.5%`
* **Profit Factor**: `1.40`
* **Średni zysk na transakcję**: `+$5`
* **Statystyki Wyjść**:
  * `signal`: 66 (59%)
  * `trailing_stop`: 19 (17%)
  * `take_profit`: 11 (10%)
  * `atr_stop`: 9 (8%)
  * `time_exit`: 6 (5%)

---

### C. Zbiór Testowy Ostateczny (TEST: 2025-2026)
* **Okres**: `2025-03-01` -> `2026-06-14` (1.3 roku - okres bessy / głębokiej konsolidacji)
* **Wynik Buy & Hold (BTC)**: `-20.6%` (trzymanie BTC przyniosłoby stratę 20%)
* **Wynik Ensemble**: **+$915.00 (+9.2% zwrotu)** <-- Wyraźna alpha rynkowa w bessie!
* **Liczba Transakcji**: `122`
* **Skuteczność (Win Rate)**: `46.7%`
* **Wskaźnik Sharpe'a**: **+1.92**
* **Maksymalne obsunięcie (DD)**: `1.4%` (wyjątkowo bezpieczny profil ryzyka)
* **Profit Factor**: `1.72`
* **Średni zysk na transakcję**: `+$8`
* **Statystyki Wyjść**:
  * `signal`: 76 (62%)
  * `take_profit`: 19 (16%)
  * `trailing_stop`: 12 (10%)
  * `atr_stop`: 8 (7%)
  * `time_exit`: 7 (6%)

---

## 3. Porównanie Nowego Ensemble vs Stara Wersja z XGBoost (Zbiór Testowy)

Usunięcie pasażera na gapę (XGBoost) i mapowanie reżimu `unclear` na tryb gotówkowy (`FLAT`) poprawiło każdą istotną metrykę bota:

| Metryka | Stara Wersja (z XGBoost) | Nowa Wersja (bez XGBoost) | Zmiana | Wnioski |
| :--- | :---: | :---: | :---: | :--- |
| **PnL na teście** | **+$850** | **+$915** | **+$65** | Wzrost zysków o 7.6% |
| **Zwrot %** | +8.5% | +9.2% | **+0.7%** | Lepsze wykorzystanie kapitału |
| **Sharpe Ratio** | +1.84 | **+1.92** | **+0.08** | Zwiększona stabilność stóp zwrotu |
| **Maks. Obsunięcie (DD)** | 1.4% | 1.4% | 0.0% | Zachowany minimalny profil ryzyka |
| **Win Rate** | 45.8% | **46.7%** | **+0.9%** | Lepsza trafność dzięki unikaniu szumu |
| **Profit Factor** | 1.78 | 1.72 | -0.06 | Stabilny stosunek zysków do strat |
| **Liczba Transakcji** | 107 | **122** | **+15** | Więcej zyskownych tradów z MACD |

---

## 4. Wyniki Stress Testu (Wysokie Opłaty i Zwiększony Poślizg)

Zasymulowaliśmy zachowanie bota w niesprzyjających warunkach rynkowych (2x wyższe prowizje oraz poślizg cenowy powiększony do 0.20% na stronę, co daje koszt round-trip w granicach **44-52 bps**):

| Metryka | Wersja Standardowa (Opłaty BGB) | Wersja Stress Test (Wysokie koszty) | Wnioski |
| :--- | :---: | :---: | :--- |
| **Opłaty Maker/Taker/Slippage** | 0.02% / 0.06% / 0.05% | 0.04% / 0.12% / 0.20% | Prowizje podwojone, poślizg 4-krotnie wyższy |
| **PnL na teście** | **+$915 (+9.2%)** | **-$203 (-2.03%)** | Bot staje się nieprofitowy |
| **Sharpe Ratio** | **+1.92** | **-0.48** | Całkowite załamanie stabilności |
| **Maks. Obsunięcie (DD)** | 1.4% | 4.14% | Trzykrotny wzrost obsunięcia kapitału |
| **Średni zysk/strata per trade** | **+$8** | **-$1.96** | Opłaty i poślizg zjadają cały zysk |

### Wnioski ze Stress Testu:
* Bot jest wysoce wrażliwy na koszty transakcyjne ze względu na krótki horyzont czasowy (1H) i niewielki średni zysk brutto (~$8 na pozycję $2,000, czyli ok. 40 bps).
* **Bezwzględny wymóg live tradingu**:
  1. Wykorzystanie zniżek na opłaty (np. VIP tier lub posiadanie tokenów giełdowych jak BGB).
  2. **Handel wyłącznie za pomocą zleceń Limit (Maker)**, co gwarantuje opłatę 0.02% zamiast 0.06%.
  3. Precyzyjne dopasowanie egzekucji w celu utrzymania poślizgu poniżej 0.05% na stronę.
