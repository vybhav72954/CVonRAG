/**
 * CVonRAG — api.js
 * Set VITE_API_URL in frontend/.env (defaults to http://localhost:8000)
 */

const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

// F8: production deploys that forget to set VITE_API_URL silently fall back to
// localhost and surface as "Cannot reach backend" with no clue why. Catch the
// misconfig at first load when the page itself is served from a non-local host.
if (typeof window !== 'undefined' && !import.meta.env.VITE_API_URL) {
  const host = window.location.hostname;
  const isLocal = host === 'localhost' || host === '127.0.0.1' || host === '0.0.0.0';
  if (!isLocal) {
    console.warn(
      `[CVonRAG] VITE_API_URL is not set; falling back to ${BASE}. ` +
      `This page is served from "${host}", so backend calls will almost certainly fail. ` +
      `Set VITE_API_URL=https://your-backend.example.com in the deploy environment.`
    );
  }
}

/** Default timeout for non-streaming requests (ms). */
const REQUEST_TIMEOUT_MS = 90_000;
/** Headers-arrival timeout for streaming requests (ms). After headers, the
 *  stream itself can run as long as the backend keeps producing events. */
const STREAM_HEADERS_TIMEOUT_MS = 120_000;

// ── Per-endpoint cancellation (F1) ───────────────────────────────────────────
// One in-flight request per endpoint family. Calling an exported function again
// aborts the prior request so its mid-flight callbacks can't pollute new state.
// Stale-event gating uses identity comparison: callbacks check whether their
// own controller is still the active one before touching the page's stores.

const active = { parse: null, recommend: null, optimize: null };

function takeOver(key) {
  active[key]?.abort();        // silent on the old request
  const c = new AbortController();
  active[key] = c;
  return c;
}

function isCurrent(key, c) {
  return active[key] === c;
}

function release(key, c) {
  if (active[key] === c) active[key] = null;
}

/**
 * Abort the in-flight request for an endpoint family if any (F4).
 * Used by the page to cancel an optimize stream when the user navigates back —
 * F1's takeOver already does this when starting a new request, but if the user
 * goes back and *doesn't* trigger a new one, the stream would otherwise keep
 * burning LLM tokens until it completes naturally.
 *
 * @param {'parse' | 'recommend' | 'optimize'} key
 */
export function abortInFlight(key) {
  active[key]?.abort();
  active[key] = null;
}

// ── Error detail normaliser (F3) ─────────────────────────────────────────────
// FastAPI returns `{detail: "..."}` for HTTPException but `{detail: [{loc, msg, ...}, ...]}`
// for Pydantic ValidationError. The page renders the message as-is, so an array
// would stringify to "[object Object],[object Object]". Flatten here.

function formatErrorDetail(body, fallback) {
  const d = body?.detail;
  if (Array.isArray(d)) {
    return d
      .map(e => {
        const loc = Array.isArray(e?.loc) ? e.loc.slice(1).join('.') : null;  // drop "body" prefix
        const msg = e?.msg ?? JSON.stringify(e);
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join('; ');
  }
  return d ?? fallback;
}

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
    // F10: caller-initiated aborts (F1's takeOver, F4's abortInFlight, browser
    // navigation) should never surface as an "interrupted" message. Real network
    // failures still bubble through.
    if (err.name === 'AbortError') return;
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
  // we clear the timer so individual events aren't cut off mid-flight.
  const controller = takeOver('parse');
  const timer = setTimeout(() => controller.abort(), STREAM_HEADERS_TIMEOUT_MS);

  let resp;
  try {
    resp = await fetch(`${BASE}/parse`, { method: 'POST', body: form, signal: controller.signal });
  } catch (err) {
    clearTimeout(timer);
    if (!isCurrent('parse', controller)) return;  // superseded → silent (F1)
    onError?.(err.name === 'AbortError'
      ? 'Parse request timed out before the backend responded.'
      : `Network error: ${err.message}`);
    release('parse', controller);
    return;
  }
  clearTimeout(timer);

  if (!resp.ok) {
    const body = await resp.json().catch(() => null);
    if (isCurrent('parse', controller)) {
      onError?.(formatErrorDetail(body, `HTTP ${resp.status} ${resp.statusText}`));
    }
    release('parse', controller);
    return;
  }

  await readSSEStream(resp, ({ type, data }) => {
    if (!isCurrent('parse', controller)) return;  // superseded → drop event (F1)
    if (type === 'progress') onProgress?.(data.data);
    if (type === 'project')  onProject?.(data.data);
    if (type === 'done')     onDone?.(data.data);
    if (type === 'error')    onError?.(data.error_message ?? 'Unknown error');
  }, msg => {
    if (!isCurrent('parse', controller)) return;
    onError?.(msg);
  });
  release('parse', controller);
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
  const controller = takeOver('recommend');
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const resp = await fetch(`${BASE}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!resp.ok) {
      const body = await resp.json().catch(() => null);
      clearTimeout(timer);
      if (isCurrent('recommend', controller)) {
        onError?.(formatErrorDetail(body, `HTTP ${resp.status} ${resp.statusText}`));
      }
      return;
    }
    const json = await resp.json();
    clearTimeout(timer);
    if (isCurrent('recommend', controller)) onDone?.(json);
  } catch (err) {
    clearTimeout(timer);
    if (!isCurrent('recommend', controller)) return;  // superseded → silent (F1)
    onError?.(err.name === 'AbortError'
      ? 'JD analysis timed out: the AI took too long. Try again or shorten the JD.'
      : err.message);
  } finally {
    release('recommend', controller);
  }
}

// ── POST /optimize — JSON → SSE bullet stream ─────────────────────────────────

/**
 * Stream optimised resume bullets.
 * @param {Object} payload - OptimizationRequest
 * @param {{ onToken?: (data: any) => void, onBullet?: (data: any) => void, onDone?: (data: any) => void, onError?: (msg: string) => void }} [callbacks]
 */
export async function optimizeResume(payload, { onToken, onBullet, onDone, onError } = {}) {
  const controller = takeOver('optimize');
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
    if (!isCurrent('optimize', controller)) return;  // superseded → silent (F1)
    onError?.(err.name === 'AbortError'
      ? 'Optimize request timed out before the backend responded.'
      : `Network error: ${err.message}`);
    release('optimize', controller);
    return;
  }
  clearTimeout(timer);

  if (!resp.ok) {
    const body = await resp.json().catch(() => null);
    if (isCurrent('optimize', controller)) {
      onError?.(formatErrorDetail(body, `HTTP ${resp.status} ${resp.statusText}`));
    }
    release('optimize', controller);
    return;
  }

  await readSSEStream(resp, ({ type, data }) => {
    if (!isCurrent('optimize', controller)) return;  // superseded → drop event (F1)
    if (type === 'token') onToken?.(data.data);
    if (type === 'bullet') onBullet?.(data.data);
    if (type === 'done') onDone?.(data.data);
    if (type === 'error') onError?.(data.error_message ?? 'Unknown error');
  }, msg => {
    if (!isCurrent('optimize', controller)) return;
    onError?.(msg);
  });
  release('optimize', controller);
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
