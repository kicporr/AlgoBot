import { useState } from 'react';
import { postAction } from '../api';
import ConfirmModal from './ConfirmModal';
import { useToast } from './Toast';

export default function PositionsTable({ positions, btcPrice }) {
  const toast = useToast();
  const [closeTarget, setCloseTarget] = useState(null);

  async function doClose() {
    if (!closeTarget) return;
    const { symbol, side } = closeTarget;
    setCloseTarget(null);
    try {
      const r = await postAction(`/close?symbol=${encodeURIComponent(symbol)}&side=${side}`);
      toast(r?.message || 'Closed', 'success');
    } catch { toast('Failed', 'error'); }
  }

  if (!positions?.length) {
    return <div className="empty">No open positions</div>;
  }

  return (
    <>
      <table>
        <thead><tr><th>Sym</th><th>Side</th><th>Size</th><th>Entry</th><th>Mark</th><th>PnL</th><th></th></tr></thead>
        <tbody>
          {positions.map((p, i) => {
            const entry = p.entry_price || 0;
            const mark = btcPrice || entry;
            const pnl = p.side === 'LONG' ? (mark - entry) * (p.size || 0) : (entry - mark) * (p.size || 0);
            const pct = entry > 0 ? ((p.side === 'LONG' ? (mark - entry) : (entry - mark)) / entry * 100) : 0;
            const sym = p.symbol ? p.symbol.split(':')[0].replace('/', '/') : '?';
            return (
              <tr key={i}>
                <td style={{ fontWeight: 600 }}>{sym}</td>
                <td style={{ color: p.side === 'LONG' ? 'var(--profit)' : 'var(--loss)' }}>{p.side.toUpperCase()}</td>
                <td>{(p.size || 0).toFixed(4)}</td>
                <td>{entry.toLocaleString()}</td>
                <td>{mark.toLocaleString()}</td>
                <td className={pnl >= 0 ? 'up' : 'down'}>
                  {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}&nbsp;({pct >= 0 ? '+' : ''}{pct.toFixed(1)}%)
                </td>
                <td>
                  <button className="btn btn-d" style={{ padding: '1px 6px', fontSize: 9 }}
                    onClick={() => setCloseTarget({ symbol: p.symbol, side: p.side?.toLowerCase() })}>x</button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {closeTarget && (
        <ConfirmModal title="Close Position"
          message={`Close ${closeTarget.symbol?.split(':')[0]?.replace('/', '/')} ${closeTarget.side?.toUpperCase()}?`}
          detail="Unrealized PnL will be realized. This cannot be undone."
          confirmLabel="Close" danger
          onConfirm={doClose} onCancel={() => setCloseTarget(null)} />
      )}
    </>
  );
}
