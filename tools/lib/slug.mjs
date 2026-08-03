// GitHub-compatible heading slugs, plus a looser "normalised" form used to
// reconcile the anchors that already exist inside the markdown sources.

// Keep letters, numbers, combining marks, connector punctuation (`_`), spaces
// and hyphens. Everything else — punctuation, emoji, arrows — is dropped, which
// is what GitHub's slugger effectively does.
const DROP = /[^\p{L}\p{N}\p{M}\p{Pc}\s-]/gu;

/** Slugify a heading the way GitHub does: lowercase, drop punctuation, spaces to hyphens. */
export function slugify(text) {
  return String(text)
    .replace(/<[^>]*>/g, '')
    .trim()
    .toLowerCase()
    .replace(DROP, '')
    .replace(/\s+/g, '-');
}

/**
 * A forgiving key for anchor lookup. Collapses runs of hyphens and trims them,
 * so `attention-mha--mqa` and `attention-mha-mqa` resolve to the same target.
 */
export function normaliseAnchor(anchor) {
  return String(anchor)
    .toLowerCase()
    .replace(/^#/, '')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

/** Slugger that appends `-1`, `-2`, … to repeated slugs, exactly like GitHub. */
export function createSlugger() {
  const seen = new Map();
  return (text) => {
    const base = slugify(text) || 'section';
    const count = seen.get(base) ?? 0;
    seen.set(base, count + 1);
    return count === 0 ? base : `${base}-${count}`;
  };
}
