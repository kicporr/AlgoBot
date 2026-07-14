import { useRef, useEffect } from 'react';

export default function StatsBar({ status }) {
  const prev = useRef({});

  useEffect(() => {
    if (!status) return;
    const curr = status.equity || 0;
    if (prev.current.equity && prev.current.equity !== curr) {
      // Trigger re-render for delta
    }
    prev.current = { equity: curr, balance: status.balance };
  }, [status]);

  if (!status) return null;
  const eq = status.equity || 0;
  const init = status.initial_capital || 10000;
  const ret = init > 0 ? ((eq - init) / init * 100) : 0;
  const prevEq = prev.current.equity;
  const eqDelta = prevEq ? (eq - prevEq).toFixed(0) : null;

  const items = [
    { label: 'Equity', value: '$' + eq.toLocaleString(undefined, { maximumFractionDigits: 0 }),
      delta: eqDelta ? (eqDelta >= 0 ? '+' : '') + eqDelta : null, cls: eq >= init ? 'val-up' : 'val-down' },
    { label: 'Return', value: (ret >= 0 ? '+' : '') + ret.toFixed(1) + '%', cls: ret >= 0 ? 'val-up' : 'val-down' },
    { label: 'Win Rate', value: (status.stats?.win_rate || 0).toFixed(1) + '%', cls: 'val-accent' },
    { label: 'Trades', value: status.stats?.total_trades || 0, cls: '' },
    { label: 'Max DD', value: (status.stats?.max_drawdown || 0).toFixed(1) + '%', cls: 'val-down' },
    { label: 'Positions', value: (status.active_positions || []).length,
      cls: (status.active_positions || []).length > 0 ? 'val-up' : '' },
    { label: 'Breaker', value: (status.circuit_breaker?.state === 'HALTED' ? 'HALTED' : 'OK'),
      cls: status.circuit_breaker?.state === 'HALTED' ? 'val-down' : 'val-up' },
    { label: 'ML Filter', value: status.meta_labeler_enabled ? 'ON' : 'OFF', cls: '' },
  ];

  return (
    <div className="stats-bar">
      {items.map(({ label, value, delta, cls }, i) => (
        <div className="stat-item" key={i}>
          <div className="stat-row">
            <span className="stat-label">{label}</span>
            {delta && <span className={`stat-delta ${parseFloat(delta) >= 0 ? 'val-up' : 'val-down'}`}>{delta}</span>}
          </div>
          <span className={`stat-value ${cls}`}>{value}</span>
          <div className="stat-bar-wrap">
            <div className={`stat-bar-fill ${cls}`} style={{ width: label === 'Win Rate' ? (status.stats?.win_rate || 0) + '%'
              : label === 'Max DD' ? Math.min((status.stats?.max_drawdown || 0) * 2, 100) + '%'
              : label === 'Return' ? Math.min(Math.abs(ret) * 2, 100) + '%'
              : '0%' }} />
          </div>
        </div>
      ))}
    </div>
  );
}
