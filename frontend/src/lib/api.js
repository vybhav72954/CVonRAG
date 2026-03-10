/**
 * CVonRAG — API client
 * Set VITE_API_URL in frontend/.env (defaults to http://localhost:8000)
 */

const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

// ── SSE frame parser ──────────────────────────────────────────────────────────

function parseSSEFrames(buffer) {
  const frames = buffer.split('\n\n');
  const remaining = frames.pop() ?? '';
  const events = [];
  for (const frame of frames) {
    if (!frame.trim()) continue;
    let eventType = 'message', dataLine = '';
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) eventType = line.slice(6).trim();
      if (line.startsWith('data:'))  dataLine  = line.slice(5).trim();
    }
    if (!dataLine) continue;
    try { events.push({ type: eventType, data: JSON.parse(dataLine) }); }
    catch { /* skip malformed */ }
  }
  return { events, remaining };
}

// ── POST /parse — file upload → SSE project stream ───────────────────────────

/**
 * Upload a .docx or .pdf and stream back parsed projects.
 * @param {File} file
 * @param {{ onProgress, onProject, onDone, onError }} callbacks
 */
export async function parseCV(file, callbacks = {}) {
  const { onProgress, onProject, onDone, onError } = callbacks;
  const form = new FormData();
  form.append('file', file);

  let resp;
  try {
    resp = await fetch(`${BASE}/parse`, { method: 'POST', body: form });
  } catch (err) {
    onError?.(`Network error: ${err.message}`);
    return;
  }
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({ detail: resp.statusText }));
    onError?.(detail.detail ?? `HTTP ${resp.status}`);
    return;
  }

  const reader  = resp.body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const { events, remaining } = parseSSEFrames(buffer);
    buffer = remaining;
    for (const { type, data } of events) {
      if (type === 'progress') onProgress?.(data);
      if (type === 'project')  onProject?.(data);
      if (type === 'done')     onDone?.(data);
      if (type === 'error')    onError?.(data.error_message ?? 'Unknown error');
    }
  }
}

// ── POST /optimize — JSON → SSE bullet stream ─────────────────────────────────

/**
 * Stream optimised resume bullets.
 * @param {Object} payload - OptimizationRequest
 * @param {{ onToken, onBullet, onDone, onError }} callbacks
 */
export async function optimizeResume(payload, callbacks = {}) {
  const { onToken, onBullet, onDone, onError } = callbacks;

  let resp;
  try {
    resp = await fetch(`${BASE}/optimize`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
  } catch (err) {
    onError?.(`Network error: ${err.message}`);
    return;
  }
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({ detail: resp.statusText }));
    onError?.(detail.detail ?? `HTTP ${resp.status}`);
    return;
  }

  const reader  = resp.body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const { events, remaining } = parseSSEFrames(buffer);
    buffer = remaining;
    for (const { type, data } of events) {
      if (type === 'token')  onToken?.(data.data);
      if (type === 'bullet') onBullet?.(data.data);
      if (type === 'done')   onDone?.(data.data);
      if (type === 'error')  onError?.(data.error_message ?? 'Unknown error');
    }
  }
}

/** GET /health */
export async function checkHealth() {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error('Health check failed');
  return r.json();
}
