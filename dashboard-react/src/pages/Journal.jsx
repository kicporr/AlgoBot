import { useState, useEffect, useMemo } from 'react';
import { fetchAllTrades } from '../api';

const SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'XRP/USDT:USDT', 'SOL/USDT:USDT', 'LTC/USDT:USDT'];

export default function Journal() {
  const [trades, setTrades] = useState([]);
  const [f, setF] = useState({ pipeline: '', symbol: '', side: '', search: '' });

  useEffect(() => { fetchAllTrades('').then(d => { if (Array.isArray(d)) setTrades(d); }); }, []);

  const filtered = useMemo(() => trades.filter(t => {
    if (f.pipeline && t.pipeline !== f.pipeline) return false;
    if (f.side && t.side !== f.side) return false;
    if (f.symbol && !(t.strategy || '').includes(f.symbol)) return false;
    if (f.search && !JSON.stringify(t).toLowerCase().includes(f.search.toLowerCase())) return false;
    return true;
  }).reverse(), [trades, f]);

  return (
    <div className="journal-wrap">
      <div className="filters">
        <select value={f.pipeline} onChange={e => setF({ ...f, pipeline: e.target.value })}>
          <option value="">All</option><option value="pure">Pure</option><option value="ml">ML</option>
        </select>
        <select value={f.symbol} onChange={e => setF({ ...f, symbol: e.target.value })}>
          <option value="">All Symbols</option>
          {SYMBOLS.map(s => <option key={s} value={s}>{s.replace(':USDT', '')}</option>)}
        </select>
        <select value={f.side} onChange={e => setF({ ...f, side: e.target.value })}>
          <option value="">All Sides</option><option value="long">Long</option><option value="short">Short</option>
        </select>
        <input placeholder="Search..." value={f.search} onChange={e => setF({ ...f, search: e.target.value })} style={{ width: 140 }} />
        <span className="dim" style={{ marginLeft: 'auto', fontFamily: 'var(--mono)', fontSize: 10 }}>{filtered.length} trades</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Exit</th><th>Sym</th><th>Side</th><th>Entry</th><th>Exit</th><th>Qty</th><th>PnL $</th><th>PnL %</th><th>Reason</th><th>Pipe</th></tr></thead>
          <tbody>
            {filtered.slice(0, 500).map((t, i) => {
              const sym = (t.strategy || '').split(':')[1]?.replace(':USDT', '') || '?';
              return (
                <tr key={i}>
                  <td className="dim">{new Date(t.exit_time || 0).toLocaleString()}</td>
                  <td style={{ fontWeight: 600 }}>{sym}</td>
                  <td style={{ color: t.side === 'long' ? 'var(--profit)' : 'var(--loss)' }}>{t.side?.toUpperCase()}</td>
                  <td>{(t.entry_price || 0).toFixed(2)}</td><td>{(t.exit_price || 0).toFixed(2)}</td>
                  <td>{(t.quantity || 0).toFixed(4)}</td>
                  <td className={(t.pnl || 0) >= 0 ? 'up' : 'down'}>{(t.pnl || 0) >= 0 ? '+' : ''}{(t.pnl || 0).toFixed(2)}</td>
                  <td className={(t.pnl || 0) >= 0 ? 'up' : 'down'}>{(t.pnl_pct || 0) >= 0 ? '+' : ''}{(t.pnl_pct || 0).toFixed(2)}%</td>
                  <td className="dim">{t.exit_reason || '?'}</td>
                  <td className="dim">{t.pipeline || '?'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
