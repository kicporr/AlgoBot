const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'journal', label: 'Journal' },
  { id: 'logs', label: 'Logs' },
  { id: 'settings', label: 'Settings' },
];

export default function Nav({ active, onChange }) {
  return (
    <nav className="nav">
      {TABS.map(t => (
        <button key={t.id} className={`nav-tab ${active===t.id?'active':''}`}
          onClick={() => onChange(t.id)}>
          {t.label}
        </button>
      ))}
    </nav>
  );
}
