export default function Header({ pipeline, onPipeline, status }) {
  const d = status || {};
  const running = d.running;
  const mode = (d.mode || 'paper').toUpperCase();
  const cb = d.circuit_breaker || {};
  const halted = cb.state === 'HALTED' || cb.manual_halted;

  return (
    <header className="header">
      <div className="header-l">
        <span className="header-brand">bocik</span>
        <div className={`header-dot ${running ? '' : 'off'}`} />
        <span className="text-mono dim" style={{ fontSize: 9 }}>{running ? 'LIVE' : 'IDLE'}</span>
        <span className="header-badge badge-mode">{mode}</span>
        <span className="text-mono dim" style={{ fontSize: 9 }}>{d.exchange || 'bitget'} {d.version || '0.1'}</span>
        {halted && <span className="header-badge badge-cb">CB HALTED</span>}
      </div>
      <div className="header-r">
        <div className="pipe-group">
          {['pure', 'ml', 'compare'].map(p => (
            <button key={p} className={`pipe-tab ${pipeline === p ? 'active' : ''}`}
              onClick={() => onPipeline(p)}>{p}</button>
          ))}
        </div>
      </div>
    </header>
  );
}
