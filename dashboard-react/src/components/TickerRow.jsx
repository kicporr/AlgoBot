import { useRef, useEffect, useState } from 'react';

export default function TickerRow({ tickers }) {
  const [, setTick] = useState(0);

  useEffect(() => {
    if (!tickers) return;
    let changed = false;
    Object.values(tickers).forEach(t => { if (t.last_price !== t._prev) changed = true; });
    if (changed) {
      setTick(n => n + 1);
      setTimeout(() => {}, 500);
    }
  }, [tickers]);

  if (!tickers) return <div className="ticker-strip" />;

  return (
    <div className="ticker-strip">
      {Object.entries(tickers).map(([sym, t]) => {
        const chg = t.change_24h_pct || 0;
        const price = t.last_price || 0;
        const name = sym.split(':')[0].replace('/', '/');
        const key = name + price;
        return (
          <div className="ticker-item ticker-flash" key={key}>
            <span className="ticker-sym">{name}</span>
            <span className="ticker-price">${price.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
            <span className="ticker-change" style={{ color: chg >= 0 ? 'var(--profit)' : 'var(--loss)' }}>
              {chg >= 0 ? '+' : ''}{chg.toFixed(2)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
