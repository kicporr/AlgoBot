import { useEffect, useRef } from 'react';

export default function ConfirmModal({ title, message, detail, confirmLabel, danger, onConfirm, onCancel }) {
  const ref = useRef(null);

  useEffect(() => {
    ref.current?.focus();
    const handler = (e) => { if (e.key === 'Escape') onCancel(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onCancel]);

  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div className="modal-box" ref={ref} tabIndex={-1} onClick={e => e.stopPropagation()}>
        <div className="modal-hd">
          <span>{title}</span>
          <button onClick={onCancel} style={{ background: 'none', border: 'none', color: 'var(--muted)', cursor: 'pointer', fontSize: 18, lineHeight: 1, padding: 0 }}>&times;</button>
        </div>
        <div className="modal-bd">
          <p>{message}</p>
          {detail && <p className="modal-detail">{detail}</p>}
        </div>
        <div className="modal-actions">
          <button className="btn" onClick={onCancel}>Cancel</button>
          <button className={`btn ${danger ? 'btn-d' : 'btn-p'}`} onClick={onConfirm}>
            {confirmLabel || 'Confirm'}
          </button>
        </div>
      </div>
    </div>
  );
}
