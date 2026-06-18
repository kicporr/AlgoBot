# Meta-Labeling: XGBoost jako filtr sygnałów MTF_MACD

## 1. Koncepcja

Meta-labeling (Lopez de Prado, 2018) — zamiast zastępować zyskowną strategię ML-em,
dodajemy drugi model który **filtruje** jej sygnały:

```
MTF_MACD.on_candle() -> Signal (LONG/SHORT/FLAT)
    |
    v
MetaLabeler.evaluate(signal, features)
    |
    +-> P(profit | features) > min_confidence -> APPROVE -> execute
    +-> else -> REJECT -> FLAT
```

**Kluczowa zaleta:** Asymetria błędów. False positive (ML odrzuca dobry sygnał)
= tracimy okazję. False negative (ML przepuszcza zły sygnał) = tracimy pieniądze.
Meta-labeler jest ostrożny — odrzuca gdy niepewny.

## 2. Architektura

### Trenowanie (offline)
```
Backtest MTF_MACD na danych historycznych (BTC, ETH, XRP, SOL, LTC)
    |
    v
Zbierz sygnały + 84 feature'y + outcome (win/loss)
    |
    v
XGBoost classifier: X = features_at_signal, y = (pnl > 0)
    |  max_depth=5, n_estimators=150, lr=0.03
    |  subsample=0.8, colsample_bytree=0.7
    |  reg_alpha=2.0, reg_lambda=3.0
    v
MetaLabeler wytrenowany -> gotowy do filtrowania
```

### Runtime (live)
```
1. MTF_MACD generuje sygnał
2. MetaLabeler.evaluate(signal, features) -> bool (~1ms)
3. Jeśli True: sygnał przechodzi do egzekucji
4. Jeśli False: sygnał downgrade'owany do FLAT (odrzucony)
5. Po zamknięciu trade'a: record_outcome() -> monitoring degradacji
```

## 3. Feature'y

84 feature'y: 65+ z FeatureEngine + 19 sygnałowych.

### Top 15 wg feature importance (XGBoost gain):

| Rank | Feature | Gain % | Kategoria | Intuicja |
|------|---------|--------|-----------|----------|
| 1 | `macd_hist_strength` | 6.9% | MACD | Siła momentum trendu |
| 2 | `bb_width_signal` | 5.0% | Bollinger | Zmienność przy sygnale |
| 3 | `bb_width` | 4.5% | Bollinger | Szerokość wstęgi |
| 4 | `bb_position_signal` | 4.5% | Bollinger | Gdzie cena w BB |
| 5 | `rsi_at_signal` | 4.0% | Momentum | RSI przy sygnale |
| 6 | `return_10` | 3.8% | Price | 10-bar momentum |
| 7 | `rsi_14` | 3.1% | Momentum | Klasyczny RSI |
| 8 | `bb_position` | 2.9% | Bollinger | Pozycja w BB |
| 9 | `di_ratio` | 2.5% | ADX/DMI | Siła trendu |
| 10 | `volatility_20` | 2.5% | Volatility | Zmienność historyczna |

### Kategorie (dobrze zdywersyfikowane):

| Kategoria | Udział |
|-----------|--------|
| Momentum (RSI, Stoch, ROC) | 17.9% |
| Bollinger (BB width, position) | 16.8% |
| Price (returns, ratios) | 11.8% |
| MACD (hist, signal) | 10.9% |
| Trend/MA (SMA, EMA, dist) | 9.8% |
| Volatility (ATR, Garman-Klass) | 7.3% |
| ADX/DMI | 6.6% |
| Volume (OBV, MFI, EOM) | 4.6% |
| Signal-specific (added) | 3.0% |
| Temporal (bars since last) | 0.6% |

**Wniosek:** Model opiera się na zrozumiałych cechach technicznych.
Top 3 feature'y (`macd_hist_strength`, `bb_width`, `bb_width_signal`) są stabilne
między oknami czasowymi (60% overlap w feature stability test).

## 4. Konfiguracja

```yaml
meta_labeling:
  enabled: true
  model: "xgboost"
  training_samples: 500      # Minimum sygnałów do treningu
  min_confidence: 0.58        # Produkcyjny próg (konserwatywny)
  max_features: 30            # Top-N feature'ów (0 = wszystkie)
```

### Dual Pipeline (MTF_MACD vs MTF_MACD + MetaLabeler)

Bot uruchamia **dwie niezależne strategie** na tych samych symbolach:
- **Pipeline "pure"**: czysty MTF_MACD, kapitał $10,000
- **Pipeline "ml"**: MTF_MACD + MetaLabeler (XGBoost filter), kapitał $10,000

Każda ma własny:
- Kapitał (balance, equity), circuit breaker, risk monitor
- Pozycje per symbol (mogą handlować tym samym symbolem jednocześnie)
- Historię transakcji i sygnałów (kolumna `pipeline` w DB)

Dashboard: przełącznik w sidebarze → wszystkie zakładki filtrowane per pipeline.

### Hiperparametry XGBoost (optymalne z grid search):

| Parametr | Wartość | Uzasadnienie |
|----------|---------|--------------|
| `max_depth` | 5 | Głębsze drzewa = lepsza interakcja cech |
| `n_estimators` | 150 | Więcej drzew przy reg_alpha=2.0 |
| `learning_rate` | 0.03 | Wolniejsze uczenie = lepsza generalizacja |
| `subsample` | 0.8 | 80% próbek na drzewo |
| `colsample_bytree` | 0.7 | 70% cech na drzewo |
| `reg_alpha` | 2.0 | L1 — redukuje overfitting |
| `reg_lambda` | 3.0 | L2 — dodatkowa regularyzacja |
| `min_child_weight` | 1 | Minimum próbek w liściu |

## 5. Wyniki testów

### Test 1 — Walk-Forward OOS (ostatnie 3 miesiące, bez retrenowania)
| | Accepted | Rejected | All |
|---|---------|----------|-----|
| Trades | 17 | 58 | 75 |
| Win Rate | **70.6%** | 25.9% | 36.0% |
| PnL | +$368 | -$565 | -$196 |
| Sharpe | 6.19 | -10.40 | -2.28 |

### Test 2 — Signal Discrimination (BTC+ETH, 60/20/20)
| | Accepted | Rejected | Baseline |
|---|---------|----------|----------|
| Trades | 71 | 167 | 238 |
| Win Rate | **74.7%** | 27.5% | 41.6% |
| PnL | +$2,395 | -$1,363 | +$1,032 |
| Sharpe | 5.13 | -5.13 | 1.91 |
| Profit Factor | 17.77 | 0.32 | 1.48 |

### Test 3 — Confidence Calibration
- Rozkład **bimodalny**: 53% predykcji <0.40, 31% >0.60, tylko 6% koło 0.50
- **Top 20% confidence → WR 94.1%**
- Bottom 20% confidence → WR 22.4%
- Model NIE jest niepewny — ma silne opinie

### Test 4 — Strict OOS (hard cutoff 2024-10-01, features osobno)
| | Accepted | Rejected | All |
|---|---------|----------|-----|
| Trades | 374 | 425 | 799 |
| Win Rate | **78.9%** | 20.0% | 47.6% |
| PnL | +$12,180 | -$5,429 | +$6,751 |
| Sharpe | 8.77 | -8.89 | 4.53 |
| Profit Factor | 9.94 | 0.18 | 1.85 |

### Test 5 — Bear Market 2022 (BTC -64.5%, trening 2020-2021)
| | Accepted | Rejected | All |
|---|---------|----------|-----|
| Trades | 261 | 271 | 532 |
| Win Rate | **78.2%** | 19.2% | 48.1% |
| PnL | +$9,036 | -$3,576 | +$5,460 |
| Sharpe | 10.16 | -9.86 | 5.52 |
| Profit Factor | 9.22 | 0.19 | 1.99 |

### Test 6 — Full Portfolio (5 symboli, 60/20/20)
| | MTF_MACD | + MetaLabeler | Zmiana |
|---|---------|---------------|--------|
| Trades | 756 | 272 | -484 (64%) |
| Win Rate | 44.6% | **83.1%** | +38.5pp |
| PnL | +$4,009 | +$9,014 | +$5,005 |
| Sharpe | 3.33 | 8.28 | +4.95 |
| Profit Factor | 1.55 | 13.90 | +12.35 |
| Max DD | 2.5% | 0.6% | -1.9pp |

### Test 7 — Feature Stability
- 3 cechy stabilne między oknami: `macd_hist_strength`, `bb_width`, `bb_width_signal`
- Overlap ratio: 60% (próg: 60%)

## 6. Monitoring degradacji (live)

| Mechanizm | Trigger | Akcja |
|-----------|---------|-------|
| **Rolling WR** | 50-trade accepted WR < 60% | Alert Telegram |
| **Monthly Sharpe** | < 2.0 przez 2 kolejne miesiące | Alert Telegram |
| **Retraining** | Co 6 miesięcy (1 sty, 1 lip) | Telegram reminder |
| **Latencja** | p99 > 100ms | Alert Telegram |
| **Latencja** | Pomiar co 50 ewaluacji | Log INFO |

## 7. Ryzyka live execution

### Latencja
- XGBoost `predict_proba()`: ~1ms na CPU
- Cały pipeline evaluate: <5ms
- Przy 1H barach — marginalne

### Degradacja modelu
- Rynek krypto zmienia strukturę co 18-24 mies.
- Model trenowany na 2020-2021 działał na 2022 (bear) i 2024-2026 (OOS)
- **Harmonogram retrenowania: co 6 miesięcy**
- Monitoring WR/Sharpe wykryje degradację przed retrainingiem

### Korelacja warunkowa
- BTC-ETH conditional PnL correlation: **0.63**
- Pozostałe pary: 0.07-0.48
- Worst day (4 concurrent positions): -$118 (1.2% DD)
- MetaLabeler naturalnie redukuje overlap (64% rejection rate)
- **Realistyczne live DD: 2-4%** (vs backtest 0.6%)

## 8. Jak odpalić

```bash
# Szybki test (BTC+ETH, 12 combo, ~2 min)
python research/meta_labeling_optimized.py --quick

# Pełny test (5 symboli, 12 combo, ~3 min)
python research/meta_labeling_optimized.py --symbols BTC,ETH,XRP,SOL,LTC --quick

# Diagnostyka overfittingu (3 testy)
python research/meta_labeling_diagnostics.py

# Głęboka optymalizacja z anti-overfitting
python research/meta_labeling_deep.py --quick

# Trenowanie na wszystkich symbolach + grid search (wolne)
python research/meta_labeling_optimized.py --symbols BTC,ETH,XRP,SOL,LTC
```

## 9. Pliki

| Plik | Rola |
|------|------|
| `strategies/meta_labeling.py` | Klasa MetaLabeler — XGBoost filtr sygnałów + monitoring degradacji |
| `research/meta_labeling_backtest.py` | v1: podstawowy backtest BTC |
| `research/meta_labeling_optimized.py` | v2: multi-symbol, grid search, signal features |
| `research/meta_labeling_deep.py` | v3: walk-forward, deflation test, feature stability |
| `research/meta_labeling_diagnostics.py` | 3 testy overfittingu (OOS, rejected, calibration) |
| `backtest/engine.py` | Dodane `features_at_signal`, `signal_type`, `regime` do trade dict |
| `orchestrator.py` | Wpięcie MetaLabeler + pomiar latencji + degradation alerts |

## 10. Revert

```bash
git checkout orchestrator.py backtest/engine.py strategies/__init__.py ml/trainer.py
rm strategies/meta_labeling.py
rm research/meta_labeling_backtest.py research/meta_labeling_optimized.py
rm research/meta_labeling_deep.py research/meta_labeling_diagnostics.py
rm docs/meta-labeling.md
```
