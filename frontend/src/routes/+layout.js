// Disable server-side rendering — this is a pure client-side app.
// adapter-static (Vercel deploy target) needs either prerender or a fallback;
// we set both: prerender=true generates index.html at build time, and the
// adapter's fallback='index.html' serves any unknown route to the same SPA
// shell. ssr=false because Svelte stores hold all state and there's nothing
// useful to render server-side.
export const ssr = false;
export const prerender = true;
