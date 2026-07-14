"""Seed demo data for dashboard development — realistic trades and positions."""

import sqlite3, os, time, random, math

DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trading.db"
)

random.seed(42)
now_ms = int(time.time() * 1000)
HOUR = 3600_000
DAY = 86_400_000

SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "XRP/USDT:USDT",
    "SOL/USDT:USDT",
    "LTC/USDT:USDT",
]
SYM_PRICES = {
    "BTC/USDT:USDT": 67500,
    "ETH/USDT:USDT": 3450,
    "XRP/USDT:USDT": 0.62,
    "SOL/USDT:USDT": 142,
    "LTC/USDT:USDT": 82,
}
SYM_NAMES = {s: s.split(":")[0].replace("/", "/") for s in SYMBOLS}

conn = sqlite3.connect(DB)
c = conn.cursor()

# Ensure trades table has pipeline column
try:
    c.execute("ALTER TABLE trades ADD COLUMN pipeline TEXT DEFAULT 'pure'")
except sqlite3.OperationalError:
    pass
try:
    c.execute("ALTER TABLE trades ADD COLUMN theoretical_entry_price REAL")
except sqlite3.OperationalError:
    pass
try:
    c.execute("ALTER TABLE trades ADD COLUMN theoretical_exit_price REAL")
except sqlite3.OperationalError:
    pass

# Clear old demo data (keep real data if any)
c.execute("DELETE FROM trades WHERE entry_time > ?", (now_ms - 90 * DAY,))
c.execute("DELETE FROM signals WHERE timestamp > ?", (now_ms - 90 * DAY,))


def generate_trade(pipeline, days_ago, symbol, side):
    """Generate a realistic trade."""
    base = SYM_PRICES[symbol]
    entry_ms = now_ms - int(days_ago * DAY) - random.randint(0, int(12 * HOUR))
    hold_hours = random.randint(3, 72)
    exit_ms = entry_ms + hold_hours * HOUR

    entry_price = base * (1 + random.uniform(-0.10, 0.10))
    is_win = random.random() < (
        0.62 if pipeline == "pure" else 0.68
    )  # ML slightly better
    if side == "long":
        exit_price = entry_price * (
            1 + random.uniform(0.005, 0.08)
            if is_win
            else 1 - random.uniform(0.005, 0.04)
        )
    else:
        exit_price = entry_price * (
            1 - random.uniform(0.005, 0.08)
            if is_win
            else 1 + random.uniform(0.005, 0.04)
        )
    quantity = round(random.uniform(0.01, 0.5), 4)
    pnl = round(
        quantity
        * entry_price
        * (
            (exit_price - entry_price) / entry_price
            if side == "long"
            else (entry_price - exit_price) / entry_price
        ),
        2,
    )
    pnl_pct = round(
        (
            (exit_price - entry_price) / entry_price * 100
            if side == "long"
            else (entry_price - exit_price) / entry_price * 100
        ),
        4,
    )

    reasons = ["take_profit", "trailing_stop", "atr_stop", "signal", "time_exit"]
    weights = [0.35, 0.30, 0.15, 0.15, 0.05]
    reason = random.choices(reasons, weights=weights, k=1)[0]

    c.execute(
        """INSERT INTO trades (entry_time, exit_time, side, entry_price, exit_price, quantity, pnl, pnl_pct, strategy, exit_reason, pipeline, theoretical_entry_price, theoretical_exit_price)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            entry_ms,
            exit_ms,
            side,
            round(entry_price, 2),
            round(exit_price, 2),
            quantity,
            pnl,
            pnl_pct,
            f"mtf_macd:{symbol}",
            reason,
            pipeline,
            round(entry_price * (1 + random.uniform(-0.001, 0.001)), 2),
            round(exit_price * (1 + random.uniform(-0.001, 0.001)), 2),
        ),
    )
    return pnl


# Generate trades for both pipelines
print("Seeding trades...")
pure_trades = []
ml_trades = []

for days in range(1, 90):
    # Each day: 0-3 trades per pipeline
    for _ in range(random.randint(0, 3)):
        sym = random.choice(SYMBOLS)
        side = random.choice(["long", "long", "short"])  # 2:1 long bias
        pnl = generate_trade("pure", days, sym, side)
        pure_trades.append(pnl)
    for _ in range(random.randint(0, 3)):
        sym = random.choice(SYMBOLS)
        side = random.choice(["long", "long", "short"])
        pnl = generate_trade("ml", days, sym, side)
        ml_trades.append(pnl)

conn.commit()

pure_total = sum(pure_trades)
pure_wins = sum(1 for p in pure_trades if p > 0)
ml_total = sum(ml_trades)
ml_wins = sum(1 for p in ml_trades if p > 0)

print(
    f"Pure: {len(pure_trades)} trades, PnL=${pure_total:.0f}, WR={pure_wins / len(pure_trades) * 100:.0f}%"
)
print(
    f"ML:   {len(ml_trades)} trades, PnL=${ml_total:.0f}, WR={ml_wins / len(ml_trades) * 100:.0f}%"
)

# Also update performance_snapshots for equity curve
try:
    c.execute("""CREATE TABLE IF NOT EXISTS performance_snapshots
        (timestamp INTEGER PRIMARY KEY, balance REAL, equity REAL, position_size REAL,
         unrealized_pnl REAL, drawdown_pct REAL, sharpe_rolling REAL)""")
except:
    pass

# Generate equity history points
eq_pure = 10000.0
eq_ml = 10000.0
for day in range(90, 0, -1):
    ts = now_ms - day * DAY
    day_pure = sum(
        p
        for p, d in zip(
            pure_trades,
            [
                t[0]
                for t in c.execute(
                    "SELECT entry_time FROM trades WHERE pipeline='pure' ORDER BY entry_time"
                ).fetchall()
            ],
        )
        if d == ts
    )
    # Simpler: just add accumulated PnL
    eq_pure = 10000 + sum(
        p for i, p in enumerate(pure_trades) if i < len(pure_trades) * (90 - day) / 90
    )
    eq_ml = 10000 + sum(
        p for i, p in enumerate(ml_trades) if i < len(ml_trades) * (90 - day) / 90
    )
    c.execute(
        "INSERT OR REPLACE INTO performance_snapshots (timestamp, balance, equity, drawdown_pct) VALUES (?,?,?,?)",
        (
            ts,
            eq_pure + eq_ml,
            eq_pure + eq_ml,
            max(0, (20000 - eq_pure - eq_ml) / 20000 * 100),
        ),
    )

conn.commit()
conn.close()

print(f"\nDone! Seeded {len(pure_trades) + len(ml_trades)} trades and equity history.")
print("Restart the bot or dashboard to see changes.")
