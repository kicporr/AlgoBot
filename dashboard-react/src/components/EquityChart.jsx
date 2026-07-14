import { useRef, useEffect } from 'react';

export default function EquityChart({ history }) {
  const canvas = useRef(null);

  useEffect(() => {
    const el = canvas.current;
    if (!el || !history?.length) return;
    const W = el.parentElement.clientWidth - 8;
    const H = 152;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    el.width = W * dpr; el.height = H * dpr;
    el.style.width = W + 'px'; el.style.height = H + 'px';
    const ctx = el.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const vals = history.map(h => h.equity || 0);
    if (vals.length < 2) return;
    const min = Math.min(...vals), max = Math.max(...vals), range = max - min || 1;

    // Grid
    ctx.strokeStyle = 'rgba(107,115,133,.06)'; ctx.lineWidth = 0.5;
    for (let i = 0; i < 4; i++) {
      const y = 4 + i * (H - 8) / 3;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Area
    const xStep = W / (vals.length - 1);
    ctx.beginPath();
    vals.forEach((v, i) => {
      const x = i * xStep;
      const y = H - ((v - min) / range) * (H - 16) - 8;
      i === 0 ? ctx.moveTo(x, H) : 0;
      ctx.lineTo(x, y);
    });
    ctx.lineTo((vals.length - 1) * xStep, H);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    const isUp = vals[vals.length - 1] >= vals[0];
    const clr = isUp ? '0,166,126' : '224,75,75';
    grad.addColorStop(0, `rgba(${clr},.12)`); grad.addColorStop(1, `rgba(${clr},0)`);
    ctx.fillStyle = grad; ctx.fill();

    // Line
    ctx.beginPath();
    ctx.strokeStyle = isUp ? '#00A67E' : '#E04B4B'; ctx.lineWidth = 1; ctx.lineJoin = 'round';
    vals.forEach((v, i) => {
      const x = i * xStep;
      const y = H - ((v - min) / range) * (H - 16) - 8;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#6B7385'; ctx.font = '9px "IBM Plex Mono", monospace';
    ctx.fillText(max.toLocaleString(undefined, { maximumFractionDigits: 0 }), 2, 12);
    ctx.fillText(min.toLocaleString(undefined, { maximumFractionDigits: 0 }), 2, H - 4);
    const last = vals[vals.length - 1];
    ctx.fillStyle = isUp ? '#00A67E' : '#E04B4B';
    ctx.fillText(last.toLocaleString(undefined, { maximumFractionDigits: 0 }), W - 60, 12);
  }, [history]);

  return <canvas ref={canvas} id="equity-canvas" />;
}
