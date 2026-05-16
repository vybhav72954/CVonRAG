import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  kit: {
    // adapter-static builds a pure-SPA bundle into ./build/ that Vercel
    // (or any static host) serves at zero compute cost. The frontend is
    // fully client-side: 3-screen wizard, all state in Svelte stores, all
    // data over `fetch` to the FastAPI backend, no +page.server.js loaders.
    // `fallback: 'index.html'` + prerender on the root layout = SPA mode.
    adapter: adapter({
      pages: 'build',
      assets: 'build',
      fallback: 'index.html',
      precompress: false,
      strict: true,
    }),
    alias: {
      $lib: 'src/lib',
    },
  },
};

export default config;
