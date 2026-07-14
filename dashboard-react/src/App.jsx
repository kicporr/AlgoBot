import { useState, useCallback, useEffect } from 'react';
import { fetchStatus } from './api';
import { ToastProvider } from './components/Toast';
import Header from './components/Header';
import Overview from './pages/Overview';
import Journal from './pages/Journal';
import Logs from './pages/Logs';
import Settings from './pages/Settings';

const TABS = { overview: Overview, journal: Journal, logs: Logs, settings: Settings };

export default function App() {
  const [pipeline, setPipeline] = useState('pure');
  const [page, setPage] = useState('overview');
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetchStatus('pure').then(setStatus).catch(() => {});
    const id = setInterval(() => fetchStatus('pure').then(setStatus).catch(() => {}), 10000);
    return () => clearInterval(id);
  }, []);

  const Page = TABS[page] || Overview;

  return (
    <ToastProvider>
      <Header pipeline={pipeline} onPipeline={setPipeline} status={status} />
      {/* Compact nav tabs */}
      <div style={{ display: 'flex', gap: 1, padding: '0 12px', background: 'var(--panel)', borderBottom: '1px solid var(--divider)', flexShrink: 0 }}>
        {Object.keys(TABS).map(k => (
          <button key={k} onClick={() => setPage(k)}
            style={{
              padding: '4px 14px', border: 'none', background: 'transparent', color: page === k ? 'var(--text)' : 'var(--muted)',
              cursor: 'pointer', fontFamily: 'var(--mono)', fontSize: 10, textTransform: 'uppercase', letterSpacing: '.3px',
              borderBottom: page === k ? '1px solid var(--accent)' : '1px solid transparent', transition: 'all .1s',
            }}>
            {k}
          </button>
        ))}
      </div>
      <Page pipeline={pipeline} />
    </ToastProvider>
  );
}
