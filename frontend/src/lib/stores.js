/**
 * CVonRAG — stores.js
 * Wizard state machine: upload → analyse → results
 *
 * Screen 1: Upload CV → parse → show all projects
 * Screen 2: Paste JD → AI recommends best projects → user confirms → generate
 * Screen 3: Streamed bullets with copy buttons
 */
import { writable, derived } from 'svelte/store';

// ── Wizard step ───────────────────────────────────────────────────────────────
/** @type {import('svelte/store').Writable<1|2|3>} */
export const step = writable(1);

// ── Step 1: parse ─────────────────────────────────────────────────────────────
export const parsedProjects = writable([]);   // all ProjectData[] from /parse
export const parseStatus    = writable('idle');  // idle|uploading|streaming|done|error
export const parseProgress  = writable('');
export const parseError     = writable('');

// ── Step 2: JD + recommendation ───────────────────────────────────────────────
export const jdText          = writable('');
export const roleType        = writable('ml_engineering');
export const charLimit       = writable(130);
export const maxBullets      = writable(2);
export const topK            = writable(3);  // how many projects to recommend

export const recommendations = writable([]);  // ProjectRecommendation[] from /recommend
export const recommendStatus = writable('idle'); // idle|loading|done|error
export const recommendError  = writable('');

// user can toggle recommended projects — starts from AI recommendation
export const selectedIds     = writable(new Set());

// ── Step 3: generation ────────────────────────────────────────────────────────
export const genStatus   = writable('idle');   // idle|streaming|done|error
export const tokenBuffer = writable('');
export const bullets     = writable([]);        // GeneratedBullet[]
export const genError    = writable('');
export const elapsed     = writable(0);

// ── Derived ───────────────────────────────────────────────────────────────────
export const isUploading    = derived(parseStatus,    s => s === 'uploading' || s === 'streaming');
export const isRecommending = derived(recommendStatus, s => s === 'loading');
export const isGenerating   = derived(genStatus,       s => s === 'streaming');

// ── Resets ────────────────────────────────────────────────────────────────────
export function resetParse() {
  parsedProjects.set([]);
  parseStatus.set('idle');
  parseProgress.set('');
  parseError.set('');
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
