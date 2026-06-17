Dobre pytanie — i dobrze, że pytasz *przed* rozpoczęciem optymalizacji, a nie po. Większość botów ginie nie przez złą strategię, tylko przez overfitting maskowany wysokim Sharpe'em in-sample.

Poniżej konkretny plan optymalizacji z zabezpieczeniami przed overfittingiem. Zakładam, że bazujemy na obecnej architekturze (MACD + reżimy + exit chain).

---

## Faza 0: Co NIE podlega optymalizacji

Zanim zaczniesz stroić, ustal co jest **zamrożone**:

| Element | Status | Powód |
|:---|:---:|:---|
| Podział walk-forward (expanding window) | ❄️ Zamrożony | Struktura walidacji musi być ustalona przed optymalizacją |
| Metryka celu (Sharpe, a nie PnL) | ❄️ Zamrożona | Optymalizacja pod PnL = przepis na overfitting |
| Max DD jako twarde ograniczenie | ❄️ Zamrożony | Minimalny akceptowalny Sharpe, maksymalne akceptowalne DD |
| Struktura reżimów (4 klasy) | ❄️ Zamrożona | Zmiana definicji reżimów = zmiana całej logiki |
| Koszty transakcyjne w backteście | ❄️ Zamrożone | Zawsze wliczone, zawsze z konserwatywnym slippage |

---

## Faza 1: Ustal hierarchię parametrów

Nie stroisz wszystkiego naraz. Parametry dzielą się na 3 warstwy według ryzyka overfittingu:

### Warstwa 1 (Niskie ryzyko — optymalizuj per-symbol)
Parametry, które mają sens ekonomiczny i są naturalnie różne między aktywami:

- **ATR multiplier dla SL** — uzasadnione różnicą w zmienności
- **Trailing stop %** — j.w.
- **Max position size (kontraktowy)** — wynika z płynności
- **Min hold bars (time exit)** — związane z cyklem świecy
- **Allow shorts (bool)** — decyzja kierunkowa, nie parametryczna

**Metoda**: Grid search z 3-5 wartościami na walk-forward. Nie ciągła optymalizacja.

### Warstwa 2 (Średnie ryzyko — optymalizuj per-symbol, z restrykcjami)
Parametry techniczne, które mogą się różnić, ale łatwo je przeoptymalizować:

- **MACD (fast, slow, signal)** — ale tylko z predefiniowanych zestawów: `(8,21,9)`, `(12,26,9)`, `(5,13,8)`, `(10,20,9)`. Nie ciągła przestrzeń.
- **Reżim ADX threshold** — z krokiem co 2 (20, 22, 24, 26, 28)
- **Reżim ATR percentile threshold** — z krokiem co 5

**Metoda**: Grid search w predefiniowanych zestawach. Klucz: **sprawdzaj stabilność** — jeśli najlepszy parametr na oknie 1 to 8/21/9, na oknie 2 to 12/26/9, na oknie 3 to 8/21/9 — to znak, że nie ma stabilnego optimum i lepiej wybrać konserwatywny default.

### Warstwa 3 (Wysokie ryzyko — globalne, NIE optymalizuj per-symbol)
Parametry, które powinny być takie same dla wszystkich aktywów:

- **Risk per trade (% kapitału)** — 20% to już agresywne, nie zwiększaj
- **Circuit breaker thresholds** (daily/weekly loss %)
- **BB period i std multiplier** dla reżimów
- **Elder filter period (daily)**

**Metoda**: Ustal raz na podstawie przekroju wszystkich symboli. Nie stroj per-asset.

---

## Faza 2: Protokół Walk-Forward do Optymalizacji

Konkretna procedura krok po kroku:

```
DLA KAŻDEGO SYMBOLU:
  Okno treningowe = pierwsze 24 miesiące
  Okno testowe   = następne 3 miesiące
  
  POWTARZAJ przesuwając okno o 3 miesiące:
    1. Optymalizuj parametry (Warstwa 1+2) TYLKO na oknie treningowym
    2. Zapisz najlepszy zestaw parametrów
    3. Testuj ten zestaw na oknie testowym (out-of-sample)
    4. ZAPISZ wyniki testowe osobno
  
  ↘ Ostateczny zestaw parametrów = mediana/średnia z najlepszych
     parametrów ze wszystkich okien treningowych
  ↘ Ostateczny Sharpe = średnia z WYNIKÓW TESTOWYCH (nie treningowych!)
```

### Test stabilności parametrów

Dla każdego parametru policz **współczynnik zmienności** (CV = std/średnia) optymalnych wartości w kolejnych oknach:

| CV | Interpretacja | Akcja |
|:---:|:---|:---|
| < 0.15 | Parametr stabilny | ✅ Użyj mediany |
| 0.15 – 0.30 | Umiarkowana zmienność | ⚠️ Użyj konserwatywnego końca zakresu |
| > 0.30 | Parametr niestabilny | 🚫 Użyj domyślnej wartości, nie optymalizuj |

---

## Faza 3: Zabezpieczenia Przed Overfittingiem

### 3a. Deflation test (test spłaszczenia Sharpe'a)

Po znalezieniu "najlepszego" zestawu parametrów, uruchom backtest na **losowo przetasowanych zwrotach** (permuted returns). Jeśli Sharpe na przetasowanych danych jest > 0.3, parametry łapią szum, nie sygnał.

### 3b. Minimum transakcji na okno testowe

Odrzuć każde okno, w którym jest **mniej niż 15 transakcji**. Mała liczba transakcji daje fałszywie zawyżony lub zaniżony Sharpe. To szczególnie ważne dla SOL.

### 3c. Reguła degradacji (od razu w kodzie backtestu)

Dla każdego okna testowego porównaj Sharpe out-of-sample do Sharpe in-sample:

```
degradacja = (Sharpe_in_sample - Sharpe_out_of_sample) / Sharpe_in_sample
```

Jeśli **degradacja > 50%** (Sharpe out-of-sample jest mniej niż połowa in-sample), to zestaw parametrów jest przeuczony — odrzuć go, nawet jeśli out-of-sample Sharpe jest dodatni.

### 3d. Test na reżimach — osobno

Nie patrz tylko na łączny Sharpe. Sprawdź wyniki w podziale na reżimy. Parametry, które dają świetny Sharpe tylko dlatego, że świetnie działają w TRENDING a fatalnie w RANGING (gdzie bot i tak jest FLAT), są OK. Ale jeśli parametry dają dobry Sharpe tylko dlatego, że TRENDING stanowił 80% okresu testowego — to pułapka. **Równowaga wyników między różnymi warunkami rynkowymi** jest ważniejsza niż pojedyncza liczba.

---

## Faza 4: Portfolio-Level Validation (to, czego teraz brakuje)

Po optymalizacji per-symbol, uruchom **symulację łącznego portfela**:

1. Połącz sygnały ze wszystkich 5 par w jedną oś czasu
2. Uwzględnij, że kapitał jest dzielony między otwarte pozycje
3. Policz **łączny Sharpe, łączne DD, łączny win rate**
4. Policz korelację zwrotów między parami w portfelu (rolling 90-day)
5. Sprawdź, czy bot nie wchodzi w >3 pozycje jednocześnie (limit koncentracji)

Kluczowa metryka: **łączny max DD portfela**. Jeśli przekracza 10%, zmniejsz `risk_per_trade` niezależnie od wyników per-asset.

---

## Faza 5: Co NIGDY nie powinno wejść do optymalizacji

- ❌ Liczba reżimów (4 to rozsądne minimum — 3 lub 5 bez silnego uzasadnienia = overfitting)
- ❌ Definicje reżimów na podstawie wyników backtestu (np. "ustawmy ADX threshold na 26 bo wtedy Sharpe rośnie o 0.3")
- ❌ Wybór aktywów na podstawie backtestu (survivorship bias — testowałeś te, które przetrwały 2020-2026)
- ❌ Optymalizacja pod konkretny okres (np. "w 2023 działało lepiej z fast=6" — nie obchodzi cię 2023, obchodzi cię przyszłość)
- ❌ Dodawanie filtrów, które eliminują pojedyncze złe transakcje z historii

---

## Plan Działania w Kolejności

| Krok | Co | Priorytet |
|:---:|:---|:---:|
| 1 | Zaimplementuj test stabilności parametrów (CV per parametr) w walk-forward | 🔴 |
| 2 | Zdefiniuj predefiniowane zestawy MACD (max 4-5) i zakresy dla pozostałych parametrów | 🔴 |
| 3 | Uruchom grid search z protokołem walk-forward dla każdego symbolu | 🔴 |
| 4 | Odrzuć parametry z CV > 0.30 i degradacją > 50% | 🔴 |
| 5 | Uruchom deflation test na finalnych parametrach | 🟡 |
| 6 | Zbuduj symulację łącznego portfela i policz skorelowane DD | 🟡 |
| 7 | Pokaż wyniki w podziale na reżimy | 🟡 |
| 8 | Podejmij decyzję o finalnym `risk_per_trade` na podstawie łącznego DD | 🟢 |

---

