import { useState, useCallback, useEffect, useRef } from 'react';
import { fetchStatus, fetchCompare, fetchRisk, postAction } from '../api';
import { useResizable } from '../hooks/useResizable';
import StatsBar from '../components/StatsBar';
import TickerRow from '../components/TickerRow';
import PositionsTable from '../components/PositionsTable';
import RecentTrades from '../components/RecentTrades';
import EquityChart from '../components/EquityChart';
import CompareView from '../components/CompareView';
import ConfirmModal from '../components/ConfirmModal';
import { useToast } from '../components/Toast';

export default function Overview({ pipeline }) {
  const [status, setStatus] = useState(null);
  const [compare, setCompare] = useState(null);
  const [risk, setRisk] = useState(null);
  const [loading, setLoading] = useState(true);
  const [modal, setModal] = useState(null);
  const prev = useRef(pipeline);
  const toast = useToast();

  // Resizable chart height (vertical)
  const chartResize = useResizable(248, 120, 500, 'vertical');
  // Resizable sidebar width (horizontal)
  const sidebarResize = useResizable(320, 220, 500, 'horizontal');

  const [chartH, setChartH] = useState(248);
  const [sidebarW, setSidebarW] = useState(320);

  useEffect(() => {
    const h = chartResize.handleRef.current;
    const s = sidebarResize.handleRef.current;
    if (h) h.addEventListener('resize', e => setChartH(e.detail));
    if (s) s.addEventListener('resize', e => setSidebarW(e.detail));
  }, []);

  useEffect(() => {
    if (prev.current !== pipeline) { setStatus(null); setRisk(null); setLoading(true); prev.current = pipeline; }
  }, [pipeline]);

  const fastPoll = useCallback(async () => {
    const s = await fetchStatus(pipeline);
    setStatus(s); setLoading(false);
  }, [pipeline]);

  const slowPoll = useCallback(async () => {
    const [c, r] = await Promise.all([fetchCompare(), fetchRisk()]);
    setCompare(c); setRisk(r);
  }, []);

  useEffect(() => {
    let c = false, fast, slow;
    const run = async () => { if (c) return; await fastPoll(); if (!c) await slowPoll(); };
    run();
    fast = setInterval(fastPoll, 1500);
    slow = setInterval(slowPoll, 5000);
    return () => { c = true; clearInterval(fast); clearInterval(slow); };
  }, [fastPoll, slowPoll]);

  async function doAction(act) {
    setModal(null);
    try { const r = await postAction(act); toast(r?.message || 'Done', 'success'); } catch { toast('Failed', 'error'); }
  }

  if (pipeline === 'compare') {
    return <div style={{ padding: 12, overflow: 'auto', height: 'calc(100vh - 100px)' }}><CompareView compare={compare} /></div>;
  }

  return (
    <>
      <TickerRow tickers={status?.tickers} />
      <StatsBar status={status} />
      <div className="chart-wrap" style={{ height: chartH }}>
        <EquityChart history={risk?.equity_history || []} />
      </div>
      <div className="drag-handle drag-handle-h" ref={chartResize.handleRef} title="Drag to resize chart" />
      <div className="grid-main" style={{ gridTemplateColumns: `1fr ${sidebarW}px` }}>
        <div>
          <div className="panel">
            <div className="panel-hd"><span>Positions</span><span className="dim">{(status?.active_positions || []).length} open</span></div>
            <div className="panel-bd">
              <PositionsTable positions={status?.active_positions} btcPrice={status?.btc_price} />
            </div>
          </div>
        </div>
        <div className="drag-handle drag-handle-v" ref={sidebarResize.handleRef} title="Drag to resize sidebar" />
        <div className="sidebar">
          <div className="side-panel">
            <div className="panel-hd">Recent Trades</div>
            <div className="panel-bd"><RecentTrades pipeline={pipeline} /></div>
          </div>
          <div className="side-panel" style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
            <button className="btn btn-a" onClick={() => setModal({ act: '/reset', title: 'Reset Circuit Breaker', msg: 'Clear all circuit breaker warnings and resume trading.', confirm: 'Reset' })}>Reset CB</button>
            <button className="btn btn-d" onClick={() => setModal({ act: '/close/all', title: 'Close All Positions', msg: 'Close ALL positions across all symbols. Unrealized PnL will be realized.', confirm: 'Close All', danger: true })}>Close All</button>
            <button className="btn btn-d" onClick={() => setModal({ act: '/emergency', title: 'Emergency Stop', msg: 'Stop the bot immediately and close all positions. This cannot be undone.', confirm: 'E-Stop', danger: true })}>E-Stop</button>
          </div>
        </div>
      </div>
      {modal && (
        <ConfirmModal title={modal.title} message={modal.msg}
          confirmLabel={modal.confirm} danger={modal.danger}
          onConfirm={() => doAction(modal.act)} onCancel={() => setModal(null)} />
      )}
    </>
  );
}
