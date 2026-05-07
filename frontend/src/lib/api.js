/**
 * CVonRAG — api.js
 * Set VITE_API_URL in frontend/.env (defaults to http://localhost:8000)
 */

const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

/** Default timeout for non-streaming requests (ms). */
const REQUEST_TIMEOUT_MS = 90_000;
/** Headers-arrival timeout for streaming requests (ms). After headers, the
 *  stream itself can run as long as the backend keeps producing events. */
const STREAM_HEADERS_TIMEOUT_MS = 120_000;

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
      if (line.startsWith('data:')) dataLine = line.slice(5).trim();
    }
    if (!dataLine) continue;
    try { events.push({ type: eventType, data: JSON.parse(dataLine) }); }
    catch { /* skip malformed SSE frame */ }
  }
  return { events, remaining };
}

/**
 * Read an SSE stream safely. Catches mid-stream disconnects and fires onError.
 * @param {Response} resp
 * @param {(event: {type: string, data: any}) => void} onEvent
 * @param {(msg: string) => void} [onError]
 */
async function readSSEStream(resp, onEvent, onError) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { events, remaining } = parseSSEFrames(buffer);
      buffer = remaining;
      for (const ev of events) onEvent(ev);
    }
  } catch (err) {
    onError?.(`Stream interrupted: ${err.message}`);
  }
}

// ── POST /parse — upload CV → SSE project stream ──────────────────────────────

/**
 * Upload a .docx or .pdf and stream back parsed projects.
 * @param {File} file
 * @param {{ onProgress?: (data: any) => void, onProject?: (data: any) => void, onDone?: (data: any) => void, onError?: (msg: string) => void }} [callbacks]
 */
export async function parseCV(file, { onProgress, onProject, onDone, onError } = {}) {
  const form = new FormData();
  form.append('file', file);

  // Timeout applies until response headers arrive. Once the SSE stream starts,
  // the abort signal is cleared so individual events aren't cut off mid-flight.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), STREAM_HEADERS_TIMEOUT_MS);

  let resp;
  try {
    resp = await fetch(`${BASE}/parse`, { method: 'POST', body: form, signal: controller.signal });
  } catch (err) {
    clearTimeout(timer);
    onError?.(err.name === 'AbortError'
      ? 'Parse request timed out before the backend responded.'
      : `Network error: ${err.message}`);
    return;
  }
  clearTimeout(timer);

  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({ detail: resp.statusText }));
    onError?.(detail.detail ?? `HTTP ${resp.status}`);
    return;
  }

  await readSSEStream(resp, ({ type, data }) => {
    if (type === 'progress') onProgress?.(data.data);
    if (type === 'project')  onProject?.(data.data);
    if (type === 'done')     onDone?.(data.data);
    if (type === 'error')    onError?.(data.error_message ?? 'Unknown error');
  }, onError);
}

// ── POST /recommend — score all projects against JD ───────────────────────────

/**
 * Ask the AI which projects are most relevant to this JD.
 * Times out after REQUEST_TIMEOUT_MS to prevent infinite hangs.
 * Uses the same callback contract as parseCV / optimizeResume.
 * @param {{ projects: Array, job_description: string, top_k: number }} payload
 * @param {{ onDone?: (data: any) => void, onError?: (msg: string) => void }} [callbacks]
 */
export async function recommendProjects(payload, { onDone, onError } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const resp = await fetch(`${BASE}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({ detail: resp.statusText }));
      onError?.(detail.detail ?? `HTTP ${resp.status}`);
      return;
    }
    onDone?.(await resp.json());
  } catch (err) {
    clearTimeout(timer);
    const msg = err.name === 'AbortError'
      ? 'JD analysis timed out: the AI took too long. Try again or shorten the JD.'
      : err.message;
    onError?.(msg);
  }
}

// ── POST /optimize — JSON → SSE bullet stream ─────────────────────────────────

/**
 * Stream optimised resume bullets.
 * @param {Object} payload - OptimizationRequest
 * @param {{ onToken?: (data: any) => void, onBullet?: (data: any) => void, onDone?: (data: any) => void, onError?: (msg: string) => void }} [callbacks]
 */
export async function optimizeResume(payload, { onToken, onBullet, onDone, onError } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), STREAM_HEADERS_TIMEOUT_MS);

  let resp;
  try {
    resp = await fetch(`${BASE}/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timer);
    onError?.(err.name === 'AbortError'
      ? 'Optimize request timed out before the backend responded.'
      : `Network error: ${err.message}`);
    return;
  }
  clearTimeout(timer);

  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({ detail: resp.statusText }));
    onError?.(detail.detail ?? `HTTP ${resp.status}`);
    return;
  }

  await readSSEStream(resp, ({ type, data }) => {
    if (type === 'token') onToken?.(data.data);
    if (type === 'bullet') onBullet?.(data.data);
    if (type === 'done') onDone?.(data.data);
    if (type === 'error') onError?.(data.error_message ?? 'Unknown error');
  }, onError);
}

/** GET /health — quick backend liveness check.
 *  Returns { ok, data?, reason? } so callers can show a banner instead of crashing. */
export async function checkHealth() {
  try {
    const r = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) return { ok: false, reason: `HTTP ${r.status}` };
    return { ok: true, data: await r.json() };
  } catch (err) {
    return {
      ok: false, reason: err.name === 'TimeoutError'
        ? 'Backend did not respond within 5 s'
        : `Cannot reach backend at ${BASE}`
    };
  }
}
