export default function StatsBar({ status }) {
  if (!status) return null;
  const eq = status.equity || 0;
  const init = status.initial_capital || 10000;
  const ret = init > 0 ? ((eq - init) / init * 100) : 0;

  const items = [
    ['Equity', '$' + eq.toLocaleString(undefined, { maximumFractionDigits: 0 }), eq >= init ? 'val-up' : 'val-down'],
    ['Return', (ret >= 0 ? '+' : '') + ret.toFixed(1) + '%', ret >= 0 ? 'val-up' : 'val-down'],
    ['Win Rate', (status.stats?.win_rate || 0).toFixed(1) + '%', 'val-accent'],
    ['Trades', status.stats?.total_trades || 0, ''],
    ['Max DD', (status.stats?.max_drawdown || 0).toFixed(1) + '%', 'val-down'],
    ['Positions', (status.active_positions || []).length, ''],
    ['Breaker', status.circuit_breaker?.state || 'OK', status.circuit_breaker?.state === 'HALTED' ? 'val-down' : 'val-up'],
    ['ML', status.meta_labeler_enabled ? 'ON' : 'OFF', ''],
  ];

  return (
    <div className="stats-bar">
      {items.map(([label, value, cls], i) => (
        <div className="stat-item" key={i}>
          <span className="stat-label">{label}</span>
          <span className={`stat-value ${cls}`}>{value}</span>
        </div>
      ))}
    </div>
  );
}
