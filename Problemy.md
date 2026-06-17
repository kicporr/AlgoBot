

## Konkretne kroki

### 1. Przetestuj na okresie bear market (2022)

Przesuń OOS window na `2022-01 → 2022-12` (crypto winter). BTC: $47k → $16k. Uruchom:
- Swój bot z tymi samymi parametrami
- Random baseline na tym samym okresie

Jeśli Twój bot ma Sharpe >0 a random <0 — to jest **dowód że reżim filter działa**. To ważniejsze niż Sharpe 2.45 w bull market.

### 2. Powiększ próbkę — dodaj 3 pozostałe symbole

2 symbole to za mało do wniosków statystycznych. Z BTC+ETH masz 1,094 trades. Z 5 symbolami będziesz miał ~2,500. To zmniejszy błąd standardowy i może przesunąć p-value poniżej 0.05.

### 3. Zastanów się nad "honest Sharpe"

W dokumentacji projektu zamiast "Sharpe 2.84" (stary, fałszywy) powinieneś napisać:

> **Realistyczny OOS Sharpe**: 2.45 (BTC+ETH, 2023-2026)
> **Random baseline**: 1.93 (średnia 1000 losowych strategii)
> **Estymowany edge**: +0.5 Sharpe'a powyżej baseline
> **Istotność statystyczna**: p ≈ 0.17 (nieistotne przy α=0.05, potrzebna większa próba)

To jest uczciwe. Inwestor, który widzi to, wie na czym stoi. Inwestor, który widzi "Sharpe 2.84" — zostaje wprowadzony w błąd.

---


