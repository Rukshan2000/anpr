export const STREAM_URL = '/api/stream';

export async function getStatus() {
  const res = await fetch('/api/status');
  if (!res.ok) throw new Error(`status ${res.status}`);
  return res.json();
}

export async function setSource(source) {
  const res = await fetch('/api/source', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source }),
  });
  if (!res.ok) throw new Error('Could not change source');
}
