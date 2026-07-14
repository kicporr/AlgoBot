import { useState, useEffect } from 'react';
import { fetchStatus, saveSettings } from '../api';

export default function Settings() {
  const [cfg, setCfg] = useState(null);
  const [mode, setMode] = useState('');
  const [saved, setSaved] = useState('');

  useEffect(() => {
    fetchStatus('pure').then(d => { if (d?.config) { setCfg(d.config); setMode(d.mode?.toLowerCase() || 'paper'); } });
  }, []);

  if (!cfg) return <div className="empty">Loading...</div>;

  function get(path, fb = '') { return path.split('.').reduce((o, k) => (o || {})[k], cfg) ?? fb; }
  function set(path, val) {
    const keys = path.split('.'); const copy = JSON.parse(JSON.stringify(cfg));
    let o = copy; for (let i = 0; i < keys.length - 1; i++) { if (!o[keys[i]]) o[keys[i]] = {}; o = o[keys[i]]; }
    o[keys[keys.length - 1]] = val; setCfg(copy);
  }

  async function save() {
    try {
      const r = await saveSettings({ risk: cfg.risk, strategies: cfg.strategies, meta_labeling: cfg.meta_labeling, mode });
      setSaved(r.status === 'ok' ? 'Saved' : 'Error');
      setTimeout(() => setSaved(''), 3000);
    } catch { setSaved('Failed'); }
  }

  const SECTIONS = [
    ['Bot', [['bot.log_level', 'Log Level', 'select', ['DEBUG', 'INFO', 'WARNING', 'ERROR']]]],
    ['Risk', [
      ['risk.max_position_pct', 'Max Position %', 'num'], ['risk.initial_capital', 'Initial Capital $', 'num'],
      ['risk.circuit_breaker.max_drawdown_pct', 'Max Drawdown %', 'num'], ['risk.circuit_breaker.daily_loss_limit_pct', 'Daily Loss Limit %', 'num'],
      ['risk.circuit_breaker.consecutive_loss_halt', 'Consecutive Losses -> Halt', 'num'],
    ]],
    ['Strategy', [
      ['strategies.mtf_macd_elder.macd.fast', 'MACD Fast', 'num'], ['strategies.mtf_macd_elder.macd.slow', 'MACD Slow', 'num'],
      ['strategies.mtf_macd_elder.macd.signal', 'MACD Signal', 'num'], ['strategies.mtf_macd_elder.exit.trailing_stop_pct', 'Trailing Stop', 'pct'],
      ['strategies.mtf_macd_elder.exit.atr_stop_mult', 'ATR Stop x', 'num'],
    ]],
    ['Meta-Labeling', [['meta_labeling.min_confidence', 'Min Confidence', 'num']]],
  ];

  return (
    <div className="settings-wrap">
      <div className="flex-between mb-12">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span className="text-mono dim" style={{ fontSize: 10 }}>Mode:</span>
          <select value={mode} onChange={e => setMode(e.target.value)} className="set-input" style={{ width: 80 }}>
            <option value="paper">paper</option><option value="live">live</option>
          </select>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {saved && <span className="text-mono" style={{ fontSize: 10, color: saved === 'Saved' ? 'var(--profit)' : 'var(--loss)' }}>{saved}</span>}
          <button className="btn btn-p" onClick={save}>Save Config</button>
        </div>
      </div>
      {SECTIONS.map(([title, fields]) => (
        <div className="section" key={title}>
          <h3>{title}</h3>
          {fields.map(([path, label, type, opts]) => {
            const raw = get(path); const val = type === 'pct' ? (raw * 100).toFixed(1) : raw;
            return (
              <div className="set-row" key={path}>
                <div><div className="set-label">{label}</div><div className="set-key">{path}</div></div>
                {type === 'select' ? (
                  <select className="set-input" value={val} onChange={e => set(path, e.target.value)}>
                    {opts.map(o => <option key={o} value={o}>{o}</option>)}
                  </select>
                ) : (
                  <input type="number" className="set-input" value={val} step="any"
                    onChange={e => set(path, type === 'pct' ? parseFloat(e.target.value || 0) / 100 : parseFloat(e.target.value || 0))} />
                )}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
