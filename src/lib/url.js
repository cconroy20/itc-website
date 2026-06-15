// Prefix a root-absolute path with the configured base path so links and
// assets resolve whether the site is served from the domain root or a subpath
// (e.g. GitHub Pages /itc-website/). Migrating is a one-line change in
// astro.config.mjs — call sites never change.
//
// url("/about")  -> "/itc-website/about"   (base="/itc-website")
// url("/about")  -> "/about"               (base="/")
// Normalize base to exactly one trailing slash so joins never double- or
// zero-out the separator regardless of how `base` is written in the config.
const BASE = import.meta.env.BASE_URL.replace(/\/+$/, "") + "/";

export function url(path = "/") {
  if (!path) return BASE;
  // Leave external links and anchors untouched.
  if (/^(https?:|mailto:|tel:|#)/.test(path)) return path;
  const clean = path.replace(/^\/+/, ""); // strip leading slashes
  return BASE + clean;
}
