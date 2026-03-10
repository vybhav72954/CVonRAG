import adapter from '@sveltejs/adapter-node';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  kit: {
    // adapter-node builds a standalone Node.js server
    // — required for Railway / Render deployment
    adapter: adapter({ out: 'build' }),
    alias: {
      $lib: 'src/lib',
    },
  },
};

export default config;
