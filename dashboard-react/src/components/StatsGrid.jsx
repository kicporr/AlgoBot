function StatCard({ label, value, cls = '', sub }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${cls}`}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

export default function StatsGrid({ status }) {
  if (!status) return null;
  const eq = status.equity || 0;
  const init = status.initial_capital || 10000;
  const retPct = init > 0 ? ((eq - init) / init * 100) : 0;

  const fmt = (n) => n.toLocaleString(undefined, {maximumFractionDigits:0});

  const items = [
    { label: 'Equity', value: '$' + fmt(eq), cls: eq >= init ? 'val-up' : 'val-down',
      sub: `Balance $${fmt(status.balance||0)}` },
    { label: 'Return', value: (retPct >= 0 ? '+' : '') + retPct.toFixed(1) + '%',
      cls: retPct >= 0 ? 'val-up' : 'val-down', sub: `Initial $${fmt(init)}` },
    { label: 'Win Rate', value: ((status.stats?.win_rate || 0)).toFixed(1) + '%', cls: 'val-info' },
    { label: 'Trades', value: status.stats?.total_trades || 0 },
    { label: 'Drawdown', value: (status.stats?.max_drawdown || 0).toFixed(1) + '%', cls: 'val-down' },
    { label: 'Positions', value: (status.active_positions||[]).length },
    { label: 'CB State', value: (status.circuit_breaker?.state || 'OK'),
      cls: status.circuit_breaker?.state === 'HALTED' ? 'val-down' : 'val-up' },
    { label: 'ML Filter', value: status.meta_labeler_enabled ? 'ON' : 'OFF',
      sub: status.meta_labeler_enabled ? 'MetaLabeler active' : 'Pure signals' },
  ];

  return (
    <div className="stats-grid">
      {items.map((it, i) => <StatCard key={i} {...it} />)}
    </div>
  );
}
