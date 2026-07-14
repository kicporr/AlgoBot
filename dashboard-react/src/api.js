const BASE = '/api';

async function fetchJSON(url) {
  const r = await fetch(BASE + url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function fetchStatus(pipeline = 'pure') {
  return fetchJSON(`/status?pipeline=${pipeline}`);
}

export async function fetchCompare() {
  return fetchJSON('/compare');
}

export async function fetchTrades(pipeline = '') {
  const q = pipeline ? `?pipeline=${pipeline}` : '';
  return fetchJSON(`/trades${q}`);
}

export async function fetchAllTrades(pipeline = '') {
  const q = pipeline ? `?pipeline=${pipeline}` : '';
  return fetchJSON(`/trades/all${q}`);
}

export async function fetchAnalytics(pipeline = '') {
  const q = pipeline ? `?pipeline=${pipeline}` : '';
  return fetchJSON(`/analytics${q}`);
}

export async function fetchLogs() {
  return fetchJSON('/logs');
}

export async function fetchRisk() {
  return fetchJSON('/risk/snapshot');
}

export async function fetchEvents(pipeline = '') {
  const q = pipeline ? `?pipeline=${pipeline}` : '';
  return fetchJSON(`/events${q}`);
}

export async function saveSettings(data) {
  const r = await fetch(BASE + '/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return r.json();
}

export async function postAction(action) {
  const r = await fetch(BASE + action, { method: 'POST' });
  return r.json();
}
