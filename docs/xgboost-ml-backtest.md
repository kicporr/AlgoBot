# XGBoost ML Backtesting

## 1. Overview

Dodatek do bocik — strategia oparta na XGBoost z cost-aware execution filter.
W przeciwieństwie do rule-based MTF_MACD, używa 65+ feature'ów technicznych do
predykcji kierunku rynku z wyprzedzeniem 4 świec (4H).

**Kluczowa idea:** Prognozuj kierunek → sprawdź czy przewaga > koszt transakcji → potwierdź momentum → wejdź.

## 2. Architektura

```
BTC/USDT 1H data (parquet) + 1D resample
  │
  ▼
FeatureEngine.bulk_compute() → 65 feature'ów (trend, momentum, volatility, volume, patterns, multi-TF)
  │
  ▼
60/20/20 split (chronologicznie)
  │
  ├─ TRAIN (60%): Grid search: λ × confidence × max_depth × lr × n_estimators
  │     │
  │     ▼
  │   BacktestEngine.run_walk_forward(XGBoostCostAware, folds=5)
  │     │
  │     per fold: strategy.retrain(train_features)
  │       ├─ build_target: UP (+1) / DOWN (0) / NOISE (dropped)
  │       │   future_return = close.shift(-4) / close - 1
  │       │   UP: return > +dead_zone  |  DOWN: return < -dead_zone
  │       │
  │       ├─ 80/20 train/val split per fold
  │       ├─ XGBClassifier.fit(X_train, y_train, eval_set=[X_val, y_val])
  │       └─ early_stopping_rounds=20
  │     │
  │     per bar: strategy.on_candle(candle, features)
  │       ├─ model.predict_proba() → prob (upward)
  │       ├─ edge = |prob - 0.5|
  │       ├─ λ_cost = lambda * (2*taker_fee + 2*slippage)
  │       ├─ if edge > λ_cost AND prob ≥ confidence_threshold:
  │       │     LONG  if ema_20_slope > 0
  │       │     SHORT if ema_20_slope < 0 AND allow_shorts
  │       └─ else: FLAT
  │
  ├─ VAL (20%): Top 10 configs → wybór najlepszego po val Sharpe
  │
  └─ TEST (20%): Ewaluacja finalna + MTF_MACD baseline + buy-hold
```

## 3. XGBoostCostAware — strategia

### Feature'y używane
Wszystkie z `FeatureEngine` **oprócz**:
- `pattern_*` — wzorce świecowe (za dużo szumu)
- `dist_sma_*`, `dist_high_*`, `dist_low_*` — dystanse do moving average (kolinearne z ceną)
- `return_5`, `return_10`, `return_20` — overlap z targetem
- `vs_4h_*`, `vs_1d_*` — multi-TF porównania (nie zawsze dostępne)
- `price`, `high_20`, `low_20`, `close_position`, `hl_ratio`, `oc_ratio`, `hl_range`, `gap`

W praktyce ~45 feature'ów trafia do modelu.

### Target
Trinary target z 4H forward return:
```
future_return = close[t+4] / close[t] - 1

UP:   return > +dead_zone_pct  → klasa 1
DOWN: return < -dead_zone_pct  → klasa 0
NOISE: |return| ≤ dead_zone   → odrzucone (szum)
```

### Cost filter
```
edge = |prob - 0.5|           # przewaga predykcyjna nad random
cost = lambda * tx_cost       # koszt round-trip × mnożnik

LONG:  prob > 0.5 + cost  AND prob ≥ confidence_threshold  AND ema_20_slope > 0
SHORT: prob < 0.5 - cost  AND (1-prob) ≥ confidence_threshold  AND ema_20_slope < 0
```

### Momentum confirmation
EMA20 slope działa jak filtr trendu — blokuje longi w trendzie spadkowym i shorty w rosnącym. Zapobiega tradingowi przeciwko momentum rynku.

## 4. Hiperparametry

| Grupa | Parametr | Default | Opis |
|-------|----------|---------|------|
| **model** | `n_estimators` | 200 | Liczba drzew |
| | `max_depth` | 5 | Głębokość drzewa |
| | `learning_rate` | 0.05 | Learning rate |
| | `subsample` | 0.7 | Fraction próbek na drzewo |
| | `colsample_bytree` | 0.6 | Fraction feature'ów na drzewo |
| | `reg_alpha` | 2.0 | L1 regularization |
| | `reg_lambda` | 3.0 | L2 regularization |
| **cost** | `lambda` | 4.0 | Mnożnik kosztów transakcyjnych |
| | `transaction_cost_bps` | 30 | Koszt round-trip w bps |
| **trading** | `confidence_threshold` | 0.55 | Minimalne prawdopodobieństwo |
| | `allow_shorts` | True | Czy zezwalać na krótkie pozycje |
| **target** | `horizon` | 4 | Forward return (bary) |
| | `dead_zone_pct` | 0.001 | Filtr szumu (0.1%) |

## 5. Jak odpalić

```bash
# Szybki test (~2 min, 25 combo)
python research/xgboost_backtest.py --quick

# Pełny grid search (~15 min, 960 combo)
python research/xgboost_backtest.py

# Na własnych danych
python research/xgboost_backtest.py --data path/to/btc_1h.parquet
```

## 6. Interpretacja wyników

Output skryptu pokazuje tabelę:

```
RESULTS: XGBoost vs MTF_MACD vs Buy-Hold
Metric               XGBoost      MTF_MACD     Buy-Hold
--------------------------------------------------------
Trades               150          180          -
Win Rate             52.3%        50.8%        -
Total PnL            $+6,200      $+1,500      $+3,000
Sharpe               2.15         1.02         -
Max DD               5.2%         3.6%         12.0%
```

**Co oznaczają flagi overfittingu:**
- `Sharpe > 3.0` — nierealistycznie wysoki, prawdopodobny overfitting
- `DD < 10%` — nierealistycznie niskie drawdown
- `train-test gap > 50%` — model nie generalizuje

**Jeśli XGBoost ma lepszego Sharpe'a od MTF_MACD** → ML działa, można rozważyć dodanie do ensemble.
**Jeśli XGBoost przegrywa** → revert, wrócić do rule-based.

## 7. Troubleshooting

| Problem | Przyczyna | Rozwiązanie |
|---------|-----------|-------------|
| 0 trades | Model nie przeszedł cost-filter | Zmniejsz `lambda`, `confidence_threshold` |
| Sharpe 0.00 | Za mało treningowych danych | Zwiększ `min_train_fraction` |
| "Insufficient training data" | Za krótki okres | Użyj dłuższego pliku parquet |
| Model not trained | Pierwszy fold bez retrain | Normalne — pierwsze N barów = FLAT |
| ValueError w retrain | Brak kolumny 'close' | Upewnij się że DataFrame ma kolumnę close |

## 8. Revert

```bash
# Przywróć stan sprzed ML
git checkout strategies/__init__.py ml/trainer.py
rm strategies/xgb_cost_aware.py
rm research/xgboost_backtest.py
rm docs/xgboost-ml-backtest.md
```
