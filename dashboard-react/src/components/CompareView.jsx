export default function CompareView({ compare }) {
  if (!compare) return <div className="empty">Loading...</div>;

  function pipeData(name, s) {
    if (!s) return null;
    const ret = s.return_pct || 0;
    const fmt = (n) => n?.toLocaleString(undefined, { maximumFractionDigits: 0 });
    return (
      <div key={name} style={{ flex: 1 }}>
        <div className="panel-hd">{name}<span className="dim">{s.meta_labeler_enabled ? 'ML' : 'pure'}</span></div>
        <div style={{ padding: 8 }}>
          <div className="stats-bar" style={{ flexWrap: 'wrap' }}>
            {[
              ['Equity', '$' + fmt(s.equity), (s.equity || 0) >= (s.initial_capital || 0) ? 'val-up' : 'val-down'],
              ['Return', (ret >= 0 ? '+' : '') + ret.toFixed(1) + '%', ret >= 0 ? 'val-up' : 'val-down'],
              ['Trades', s.trade_count || 0, ''],
              ['Win Rate', (s.win_rate || 0).toFixed(1) + '%', 'val-accent'],
              ['Active', s.active_positions || 0, ''],
              ['Unrealized', '$' + (s.unrealized_pnl || 0).toFixed(0), (s.unrealized_pnl || 0) >= 0 ? 'val-up' : 'val-down'],
            ].map(([l, v, c]) => (
              <div className="stat-item" key={l} style={{ minWidth: 100 }}>
                <span className="stat-label">{l}</span>
                <span className={`stat-value ${c}`} style={{ fontSize: 14 }}>{v}</span>
              </div>
            ))}
          </div>
          <div className="text-mono dim" style={{ fontSize: 9, marginTop: 8 }}>
            CB: {s.circuit_breaker || 'OK'} · Balance: ${fmt(s.balance)} · Capital: ${fmt(s.initial_capital)}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', gap: 1, background: 'var(--border)', height: '100%' }}>
      {pipeData('PURE', compare.pure)}
      {pipeData('ML', compare.ml)}
      {[pipeData('PURE', compare.pure), pipeData('ML', compare.ml)].filter(Boolean).length === 0 && <div className="empty">No pipeline data</div>}
    </div>
  );
}
