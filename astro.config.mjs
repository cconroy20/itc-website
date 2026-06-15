// @ts-check
import { defineConfig } from 'astro/config';

// --- Deployment target -------------------------------------------------------
// To migrate to a permanent home, change ONLY these two values:
//   • Root domain (e.g. https://itc.cfa.harvard.edu):  SITE = '...', BASE = '/'
//   • A different GitHub project repo:                 BASE = '/<repo-name>'
// Everything else (asset URLs, nav links) derives from BASE automatically
// because templates build links with `import.meta.env.BASE_URL`.
const SITE = 'https://cconroy20.github.io';
const BASE = '/itc-website';
// -----------------------------------------------------------------------------

// https://astro.build/config
export default defineConfig({
  site: SITE,
  base: BASE,
});
