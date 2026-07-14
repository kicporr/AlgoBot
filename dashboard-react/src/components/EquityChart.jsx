import { useRef, useEffect, useState } from 'react';

export default function EquityChart({ history }) {
  const canvas = useRef(null);
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    const el = canvas.current;
    if (!el || !history?.length) return;

    const W = el.parentElement.clientWidth - 16;
    const H = 240;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    el.width = W * dpr; el.height = H * dpr;
    el.style.width = W + 'px'; el.style.height = H + 'px';
    const ctx = el.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const vals = history.map(h => h.equity || 0);
    if (vals.length < 2) return;

    const pad = { top: 28, bot: 24, left: 8, right: 8 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bot;
    const min = Math.min(...vals), max = Math.max(...vals), range = max - min || 1;

    // Background
    ctx.fillStyle = '#0E1117'; ctx.fillRect(0, 0, W, H);

    // Horizontal grid
    const gridLines = 5;
    ctx.strokeStyle = 'rgba(107,115,133,.06)'; ctx.lineWidth = 0.5;
    ctx.fillStyle = '#6B7385'; ctx.font = '9px "IBM Plex Mono", monospace';
    for (let i = 0; i <= gridLines; i++) {
      const y = pad.top + (plotH / gridLines) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
      const val = max - (range / gridLines) * i;
      ctx.fillText('$' + val.toLocaleString(undefined, { maximumFractionDigits: 0 }), pad.left + 2, y - 4);
    }

    // Vertical time grid
    ctx.strokeStyle = 'rgba(107,115,133,.04)';
    const vLines = 6;
    for (let i = 0; i <= vLines; i++) {
      const x = pad.left + (plotW / vLines) * i;
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, H - pad.bot); ctx.stroke();
    }

    // Clip area for chart
    ctx.save();
    ctx.beginPath(); ctx.rect(pad.left, pad.top, plotW, plotH); ctx.clip();

    const xStep = plotW / (vals.length - 1);

    // Area fill
    const isUp = vals[vals.length - 1] >= vals[0];
    const clr = isUp ? '0,166,126' : '224,75,75';
    ctx.beginPath();
    vals.forEach((v, i) => {
      const x = pad.left + i * xStep;
      const y = pad.top + plotH - ((v - min) / range) * plotH;
      i === 0 ? ctx.moveTo(x, pad.top + plotH) : 0;
      ctx.lineTo(x, y);
    });
    ctx.lineTo(pad.left + (vals.length - 1) * xStep, pad.top + plotH);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    grad.addColorStop(0, `rgba(${clr},.20)`); grad.addColorStop(0.6, `rgba(${clr},.04)`); grad.addColorStop(1, `rgba(${clr},0)`);
    ctx.fillStyle = grad; ctx.fill();

    // Draw line
    ctx.beginPath();
    ctx.strokeStyle = isUp ? '#00A67E' : '#E04B4B'; ctx.lineWidth = 1.2; ctx.lineJoin = 'round';
    vals.forEach((v, i) => {
      const x = pad.left + i * xStep;
      const y = pad.top + plotH - ((v - min) / range) * plotH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Start dot
    const startX = pad.left, startY = pad.top + plotH - ((vals[0] - min) / range) * plotH;
    ctx.beginPath(); ctx.arc(startX, startY, 2.5, 0, Math.PI * 2); ctx.fillStyle = isUp ? '#00A67E' : '#E04B4B'; ctx.fill();

    // End dot
    const endX = pad.left + (vals.length - 1) * xStep, endY = pad.top + plotH - ((vals[vals.length - 1] - min) / range) * plotH;
    ctx.beginPath(); ctx.arc(endX, endY, 3, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill();
    ctx.beginPath(); ctx.arc(endX, endY, 2, 0, Math.PI * 2); ctx.fillStyle = isUp ? '#00A67E' : '#E04B4B'; ctx.fill();

    ctx.restore();

    // End label
    const lastVal = vals[vals.length - 1];
    ctx.fillStyle = '#CDD6E0'; ctx.font = '600 13px "IBM Plex Mono", monospace';
    const labelW = ctx.measureText('$' + lastVal.toLocaleString(undefined, { maximumFractionDigits: 0 })).width;
    const labelX = Math.min(endX + 8, W - labelW - 4);
    ctx.fillText('$' + lastVal.toLocaleString(undefined, { maximumFractionDigits: 0 }), labelX, endY + 4);

    // Hover handler
    const rect = el.getBoundingClientRect();
    const onMove = (e) => {
      const mx = (e.clientX - rect.left) * (W / rect.width) - pad.left;
      if (mx < 0 || mx > plotW) { setTooltip(null); return; }
      const idx = Math.round(mx / xStep);
      if (idx >= 0 && idx < vals.length) {
        setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, value: vals[idx], index: idx, total: vals.length });
      }
    };
    const onLeave = () => setTooltip(null);
    el.addEventListener('mousemove', onMove);
    el.addEventListener('mouseleave', onLeave);
    return () => { el.removeEventListener('mousemove', onMove); el.removeEventListener('mouseleave', onLeave); };
  }, [history]);

  return (
    <div style={{ position: 'relative' }}>
      <canvas ref={canvas} style={{ cursor: 'crosshair' }} />
      {tooltip && (
        <div style={{
          position: 'absolute', left: Math.min(tooltip.x + 10, (canvas.current?.parentElement?.clientWidth || 400) - 140),
          top: tooltip.y - 36, pointerEvents: 'none', zIndex: 10,
          background: 'var(--panel)', border: '1px solid var(--border)', padding: '4px 8px',
          fontFamily: 'var(--mono)', fontSize: 10,
        }}>
          ${tooltip.value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          <span className="dim" style={{ marginLeft: 6 }}>{tooltip.index + 1}/{tooltip.total}</span>
        </div>
      )}
    </div>
  );
}
