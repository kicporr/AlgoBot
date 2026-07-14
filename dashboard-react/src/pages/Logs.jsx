import { useState, useEffect, useCallback, useMemo } from 'react';
import { fetchLogs } from '../api';

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
const LVL = { ERROR: 'E', WARNING: 'W', INFO: 'I', DEBUG: 'D' };

export default function Logs() {
  const [logs, setLogs] = useState([]);
  const [level, setLevel] = useState('');
  const [search, setSearch] = useState('');

  const load = useCallback(() => { fetchLogs().then(d => { if (Array.isArray(d)) setLogs(d); }); }, []);
  useEffect(() => { load(); const id = setInterval(load, 5000); return () => clearInterval(id); }, [load]);

  const filtered = useMemo(() => logs.filter(l => {
    if (level && l.level !== level) return false;
    if (search && !l.message.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  }), [logs, level, search]);

  return (
    <div className="journal-wrap">
      <div className="filters">
        <select value={level} onChange={e => setLevel(e.target.value)}>
          <option value="">All</option>
          {Object.entries(LVL).map(([k, v]) => <option key={k} value={k}>{v} {k}</option>)}
        </select>
        <input placeholder="Filter..." value={search} onChange={e => setSearch(e.target.value)} style={{ width: 180 }} />
        <span className="dim" style={{ marginLeft: 'auto', fontFamily: 'var(--mono)', fontSize: 10 }}>{filtered.length} lines</span>
      </div>
      <div className="table-wrap">
        {filtered.map((l, i) => (
          <div className="log-line" key={i}>
            <span className="log-ts">{l.timestamp || ''}</span>
            <span className={`log-lvl lvl-${LVL[l.level] || 'D'}`}>{LVL[l.level] || 'D'}</span>
            <span className="log-msg">{esc(l.message)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
