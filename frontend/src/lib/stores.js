/**
 * CVonRAG — stores.js
 * Wizard state machine: upload → analyse → results
 *
 * Screen 1: Upload CV → parse → show all projects
 * Screen 2: Paste JD → AI recommends best projects → user confirms → generate
 * Screen 3: Streamed bullets with copy buttons
 *
 * Persistence: long-lived state (parsed projects, JD, settings, completed
 * bullets, current step) is mirrored to sessionStorage so a browser refresh
 * mid-flow doesn't dump the user back to step 1. Transient state (status
 * flags, progress messages, in-flight token buffer, errors) stays in-memory
 * — it has no meaning after a reload and would render a misleading UI
 * (e.g., a "streaming…" banner with no actual stream behind it).
 *
 * sessionStorage (not localStorage): scoped to the tab so multiple tabs get
 * independent wizards, and state auto-clears when the tab closes.
 */
import { writable, derived } from 'svelte/store';

// ── sessionStorage-backed writable ───────────────────────────────────────────

const STORAGE_PREFIX = 'cvonrag:';
const isBrowser =
  typeof window !== 'undefined' && typeof window.sessionStorage !== 'undefined';

function _read(key, fallback) {
  if (!isBrowser) return fallback;
  try {
    const raw = window.sessionStorage.getItem(STORAGE_PREFIX + key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function _write(key, value) {
  if (!isBrowser) return;
  try {
    window.sessionStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(value));
  } catch {
    /* quota exceeded or private mode — silently degrade to in-memory only */
  }
}

/**
 * Writable store backed by sessionStorage. Survives browser refresh; resets
 * when the tab closes. `serialize` / `deserialize` let non-JSON types (e.g.
 * Set) round-trip through storage.
 */
function persistent(key, initial, { serialize, deserialize } = {}) {
  const stored = _read(key, undefined);
  const start = stored === undefined
    ? initial
    : (deserialize ? deserialize(stored) : stored);
  const store = writable(start);
  if (isBrowser) {
    store.subscribe(value => _write(key, serialize ? serialize(value) : value));
  }
  return store;
}

// Sets don't JSON-serialize natively → store as Array, restore as Set.
const setHelpers = {
  serialize:   (s) => Array.from(s ?? []),
  deserialize: (a) => new Set(Array.isArray(a) ? a : []),
};

// ── Wizard step ───────────────────────────────────────────────────────────────
/** @type {import('svelte/store').Writable<1|2|3>} */
export const step = persistent('step', 1);

// ── Step 1: parse ─────────────────────────────────────────────────────────────
export const parsedProjects = persistent('parsedProjects', []);  // /parse output
export const parseStatus    = writable('idle');  // transient: idle|uploading|streaming|done|error
export const parseProgress  = writable('');      // transient
export const parseError     = writable('');      // transient
// Per-project errors (F2): the backend's /parse stream emits one `error` event
// per failed project then continues with the rest. Without a separate channel,
// the final `done` would flip parseStatus from 'error' to 'done' and the user
// would never see the partial-failure messages. parseWarnings is rendered
// independently of status so the messages survive the transition.
export const parseWarnings  = writable([]);      // transient

// ── Step 2: JD + recommendation ───────────────────────────────────────────────
export const jdText          = persistent('jdText', '');
export const roleType        = persistent('roleType', 'ml_engineering');
export const charLimit       = persistent('charLimit', 130);
export const maxBullets      = persistent('maxBullets', 2);
export const topK            = persistent('topK', 3);  // how many projects to recommend

export const recommendations = persistent('recommendations', []);  // /recommend output
export const recommendStatus = writable('idle');  // transient: idle|loading|done|error
export const recommendError  = writable('');      // transient

// user can toggle recommended projects — starts from AI recommendation
export const selectedIds     = persistent('selectedIds', new Set(), setHelpers);

// ── Step 3: generation ────────────────────────────────────────────────────────
// genStatus is transient: a refresh kills the in-flight stream regardless,
// so we always boot back to 'idle' and let the user re-trigger if they want
// more bullets. Already-completed bullets persist so the user keeps their work.
export const genStatus   = writable('idle');   // transient: idle|streaming|done|error
export const tokenBuffer = writable('');       // transient: only meaningful mid-stream
export const bullets     = persistent('bullets', []);  // GeneratedBullet[]
export const genError    = writable('');       // transient
export const elapsed     = persistent('elapsed', 0);

// ── Derived ───────────────────────────────────────────────────────────────────
export const isUploading    = derived(parseStatus,    s => s === 'uploading' || s === 'streaming');
export const isRecommending = derived(recommendStatus, s => s === 'loading');
export const isGenerating   = derived(genStatus,       s => s === 'streaming');

// ── Resets ────────────────────────────────────────────────────────────────────
// Each .set() also overwrites sessionStorage via the persistent subscriber,
// so calling these clears persisted state too — no separate purge needed.
export function resetParse() {
  parsedProjects.set([]);
  parseStatus.set('idle');
  parseProgress.set('');
  parseError.set('');
  parseWarnings.set([]);
}

export function resetRecommend() {
  recommendations.set([]);
  recommendStatus.set('idle');
  recommendError.set('');
  selectedIds.set(new Set());
}

export function resetGeneration() {
  genStatus.set('idle');
  tokenBuffer.set('');
  bullets.set([]);
  genError.set('');
  elapsed.set(0);
}
