import { useEffect, useRef, useState } from 'react';

export function usePolling(callback, intervalMs, deps = []) {
  const saved = useRef(callback);
  const timer = useRef(null);
  const mounted = useRef(true);
  const [loading, setLoading] = useState(true);
  saved.current = callback;

  useEffect(() => {
    mounted.current = true;
    setLoading(true);

    const tick = async () => {
      if (!mounted.current) return;
      await saved.current();
      if (mounted.current) setLoading(false);
    };

    // Fire immediately
    tick();

    // Then poll
    timer.current = setInterval(tick, intervalMs);

    return () => {
      mounted.current = false;
      clearInterval(timer.current);
    };
  }, [intervalMs, ...deps]);

  return { loading };
}
