import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  kit: {
    // adapter-static builds a pure-SPA bundle into ./build/ that Vercel
    // (or any static host) serves at zero compute cost. The frontend is
    // fully client-side: 3-screen wizard, all state in Svelte stores, all
    // data over `fetch` to the FastAPI backend, no +page.server.js loaders.
    // No `fallback` — the app is single-route (`+page.svelte` at `/`) and
    // `prerender = true` on the root layout generates `build/index.html`
    // directly. Setting `fallback: 'index.html'` previously caused a build
    // warning ("Overwriting build/index.html with fallback page") because
    // the fallback file would clobber the prerendered one — same SPA shell
    // content either way (ssr=false), but the overwrite is noise. If a
    // future change adds client-side routing for unprerendered paths,
    // re-introduce `fallback: '200.html'` and add a Vercel rewrite.
    adapter: adapter({
      pages: 'build',
      assets: 'build',
      precompress: false,
      strict: true,
    }),
    alias: {
      $lib: 'src/lib',
    },
  },
};

export default config;
