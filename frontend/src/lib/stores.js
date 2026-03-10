/**
 * CVonRAG — Svelte stores
 * Wizard state machine: upload → jd → results
 */
import { writable, derived } from 'svelte/store';

// ── Wizard navigation ─────────────────────────────────────────────────────────
/** @type {import('svelte/store').Writable<1|2|3>} */
export const step = writable(1);

// ── Step 1: parsed projects ───────────────────────────────────────────────────
/** @type {import('svelte/store').Writable<Array>} */
export const parsedProjects = writable([]);    // full ProjectData[]
export const selectedIds    = writable(new Set()); // Set of project_ids
export const parseStatus    = writable('idle');    // idle|uploading|streaming|done|error
export const parseProgress  = writable('');        // latest progress message
export const parseError     = writable('');

// ── Step 2: JD + settings ─────────────────────────────────────────────────────
export const jdText      = writable('');
export const roleType    = writable('ml_engineering');
export const charLimit   = writable(130);
export const maxBullets  = writable(2);

// ── Step 3: generation ────────────────────────────────────────────────────────
export const genStatus   = writable('idle');   // idle|streaming|done|error
export const tokenBuffer = writable('');       // live typewriter text
export const bullets     = writable([]);       // GeneratedBullet[]
export const genError    = writable('');
export const elapsed     = writable(0);

// ── Derived ───────────────────────────────────────────────────────────────────
export const isUploading  = derived(parseStatus, s => s === 'uploading' || s === 'streaming');
export const isGenerating = derived(genStatus,   s => s === 'streaming');
export const hasResults   = derived(bullets,     b => b.length > 0);

// ── Reset helpers ─────────────────────────────────────────────────────────────
export function resetParse() {
  parsedProjects.set([]);
  selectedIds.set(new Set());
  parseStatus.set('idle');
  parseProgress.set('');
  parseError.set('');
}

export function resetGeneration() {
  genStatus.set('idle');
  tokenBuffer.set('');
  bullets.set([]);
  genError.set('');
  elapsed.set(0);
}
