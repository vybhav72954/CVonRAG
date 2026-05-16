// Disable server-side rendering — this is a pure client-side app.
// adapter-static (Vercel deploy target) needs either prerender or a fallback
// per route. The app is single-route (`+page.svelte` at `/`), so prerender
// alone is sufficient — it generates `build/index.html` from the Svelte
// shell. ssr=false because Svelte stores hold all state and there's nothing
// useful to render server-side. No adapter fallback is configured;
// unprerendered paths 404 (acceptable since there are none).
export const ssr = false;
export const prerender = true;
