import { useState, useEffect } from 'react';
import { fetchTrades } from '../api';

export default function RecentTrades({ pipeline }) {
  const [trades, setTrades] = useState([]);

  useEffect(() => {
    let c = false;
    const load = () => fetchTrades(pipeline).then(d => { if (!c && Array.isArray(d)) setTrades(d.slice(0, 15)); });
    load(); const id = setInterval(load, 5000);
    return () => { c = true; clearInterval(id); };
  }, [pipeline]);

  if (!trades.length) return <div className="empty">No trades yet</div>;

  return (
    <table>
      <thead><tr><th>Time</th><th>Sym</th><th>Side</th><th>PnL</th><th>Reason</th></tr></thead>
      <tbody>
        {trades.map((t, i) => {
          const sym = (t.strategy || '').split(':')[1]?.split(':')[0]?.replace('/', '/') || '?';
          const time = new Date(t.exit_time || 0).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          return (
            <tr key={i}>
              <td className="dim">{time}</td>
              <td style={{ fontWeight: 600 }}>{sym}</td>
              <td style={{ color: t.side === 'long' ? 'var(--profit)' : 'var(--loss)' }}>{t.side?.toUpperCase()}</td>
              <td className={(t.pnl || 0) >= 0 ? 'up' : 'down'}>{(t.pnl || 0) >= 0 ? '+' : ''}${(t.pnl || 0).toFixed(0)}</td>
              <td className="dim">{t.exit_reason || '?'}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
