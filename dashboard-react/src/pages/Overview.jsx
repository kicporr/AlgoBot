import { useState, useCallback, useEffect, useRef } from 'react';
import { fetchStatus, fetchCompare, fetchRisk } from '../api';
import StatsGrid from '../components/StatsGrid';
import TickerRow from '../components/TickerRow';
import PositionsTable from '../components/PositionsTable';
import RecentTrades from '../components/RecentTrades';
import EquityChart from '../components/EquityChart';
import CompareView from '../components/CompareView';

function Skeleton({ h = 200 }) {
  return <div className="skeleton" style={{ height: h }} />;
}

export default function Overview({ pipeline }) {
  const [status, setStatus] = useState(null);
  const [compare, setCompare] = useState(null);
  const [risk, setRisk] = useState(null);
  const [loading, setLoading] = useState(true);
  const prevPipeline = useRef(pipeline);

  // Clear stale data instantly on pipeline switch
  useEffect(() => {
    if (prevPipeline.current !== pipeline) {
      setStatus(null);
      setCompare(null);
      setRisk(null);
      setLoading(true);
      prevPipeline.current = pipeline;
    }
  }, [pipeline]);

  // Fast poll (positions, tickers): 1.5s
  const fastPoll = useCallback(async () => {
    const s = await fetchStatus(pipeline);
    setStatus(s);
    setLoading(false);
  }, [pipeline]);

  // Slow poll (analytics, equity): 5s
  const slowPoll = useCallback(async () => {
    const [c, r] = await Promise.all([fetchCompare(), fetchRisk()]);
    setCompare(c); setRisk(r);
  }, []);

  useEffect(() => {
    let fastId, slowId;
    let cancelled = false;

    const run = async () => {
      if (cancelled) return;
      await fastPoll();
      if (cancelled) return;
      await slowPoll();
    };
    run(); // First fetch: both, immediately

    fastId = setInterval(fastPoll, 1500);
    slowId = setInterval(slowPoll, 5000);

    return () => { cancelled = true; clearInterval(fastId); clearInterval(slowId); };
  }, [fastPoll, slowPoll]);

  if (pipeline === 'compare') {
    return <CompareView compare={compare} />;
  }

  return (
    <div className={`page-transition ${loading ? 'loading' : ''}`}>
      {loading && !status ? (
        <>
          <div className="stats-grid">{Array.from({ length: 8 }, (_, i) => <Skeleton key={i} h={76} />)}</div>
          <Skeleton h={60} />
        </>
      ) : (
        <>
          <StatsGrid status={status} />
          <TickerRow tickers={status?.tickers} />
        </>
      )}

      <div className="grid-2 mb-12">
        <div className="panel">
          <div className="panel-header">
            Positions <span className="dim">{(status?.active_positions || []).length} open</span>
          </div>
          <div className="panel-body">
            {loading && !status ? <Skeleton h={100} /> : <PositionsTable positions={status?.active_positions} btcPrice={status?.btc_price} />}
          </div>
        </div>
        <div className="panel">
          <div className="panel-header">Recent Trades</div>
          <div className="panel-body">
            <RecentTrades pipeline={pipeline} />
          </div>
        </div>
      </div>

      <div className="panel mb-16">
        <div className="panel-header">Equity Curve</div>
        <div className="panel-body">
          {loading && !risk ? <Skeleton h={280} /> : <EquityChart history={risk?.equity_history || []} />}
        </div>
      </div>
    </div>
  );
}
