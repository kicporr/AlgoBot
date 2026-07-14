import { useRef, useEffect, useState } from 'react';

export default function TickerRow({ tickers }) {
  const prevPrices = useRef({});
  const [, force] = useState(0);

  useEffect(() => {
    if (!tickers) return;
    let changed = false;
    Object.entries(tickers).forEach(([sym, t]) => {
      const prev = prevPrices.current[sym];
      if (prev !== undefined && prev !== t.last_price) changed = true;
      prevPrices.current[sym] = t.last_price;
    });
    if (changed) force(n => n + 1);
  }, [tickers]);

  if (!tickers) return <div className="ticker-strip" />;

  return (
    <div className="ticker-strip">
      {Object.entries(tickers).map(([sym, t], idx) => {
        const chg = t.change_24h_pct || 0;
        const price = t.last_price || 0;
        const name = sym.split(':')[0].split('/')[0];
        const name2 = sym.split(':')[0].split('/')[1] || '';
        const prev = prevPrices.current[sym];
        const flashed = prev !== undefined && prev !== price;
        const isUp = price >= (prev || price);

        return (
          <div className={`ticker-item ${flashed ? 'ticker-flash' : ''}`} key={sym}
            style={{ background: idx % 2 === 0 ? 'transparent' : 'rgba(107,115,133,.02)' }}>
            <span className="ticker-sym">{name}</span>
            <span className="ticker-quote">{name2}</span>
            <span className="ticker-price">{price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
            <span className="ticker-arrow" style={{ color: chg >= 0 ? 'var(--profit)' : 'var(--loss)' }}>
              {chg >= 0 ? '▲' : '▼'}
            </span>
            <span className="ticker-change" style={{ color: chg >= 0 ? 'var(--profit)' : 'var(--loss)' }}>
              {chg >= 0 ? '+' : ''}{chg.toFixed(2)}%
            </span>
            {flashed && (
              <span className="ticker-delta" style={{ color: isUp ? 'var(--profit)' : 'var(--loss)' }}>
                {isUp ? '+' : '-'}{Math.abs(price - prev).toFixed(1)}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
