// The markdown → HTML pipeline.
//
// Three things happen beyond a plain render:
//   1. code is highlighted at build time (no scripts, no CDN, no flash),
//   2. the guides' recurring markers (⚠ Trap, 🗣 Say this in the room, …)
//      become styled callouts,
//   3. links between the markdown files are rewritten to site URLs.

import MarkdownIt from 'markdown-it';
import hljs from 'highlight.js/lib/core';

import bash from 'highlight.js/lib/languages/bash';
import c from 'highlight.js/lib/languages/c';
import dockerfile from 'highlight.js/lib/languages/dockerfile';
import ini from 'highlight.js/lib/languages/ini';
import javascript from 'highlight.js/lib/languages/javascript';
import json from 'highlight.js/lib/languages/json';
import lua from 'highlight.js/lib/languages/lua';
import makefile from 'highlight.js/lib/languages/makefile';
import markdownLang from 'highlight.js/lib/languages/markdown';
import nginx from 'highlight.js/lib/languages/nginx';
import plaintext from 'highlight.js/lib/languages/plaintext';
import protobuf from 'highlight.js/lib/languages/protobuf';
import python from 'highlight.js/lib/languages/python';
import rust from 'highlight.js/lib/languages/rust';
import sql from 'highlight.js/lib/languages/sql';
import typescript from 'highlight.js/lib/languages/typescript';
import xml from 'highlight.js/lib/languages/xml';
import yaml from 'highlight.js/lib/languages/yaml';

import { createSlugger, normaliseAnchor } from './slug.mjs';

for (const [name, language] of Object.entries({
  bash, c, dockerfile, ini, javascript, json, lua, makefile,
  markdown: markdownLang, nginx, plaintext, protobuf, python, rust, sql,
  typescript, xml, yaml,
})) {
  hljs.registerLanguage(name, language);
}

const LANGUAGE_ALIASES = {
  sh: 'bash', shell: 'bash', console: 'bash', zsh: 'bash',
  toml: 'ini', cfg: 'ini', conf: 'ini',
  jsonc: 'json', json5: 'json',
  py: 'python', proto: 'protobuf', make: 'makefile',
  js: 'javascript', ts: 'typescript',
  html: 'xml', jinja: 'xml',
  yml: 'yaml',
  text: 'plaintext', txt: 'plaintext', promql: 'plaintext', colang: 'plaintext',
  psql: 'sql', postgresql: 'sql',
};

/** Marker → callout kind. The emoji are the guides' own vocabulary. */
const CALLOUTS = [
  ['⚠', 'trap'],
  ['🗣', 'say'],
  ['📐', 'numbers'],
  ['📄', 'paper'],
  ['💰', 'math'],
  ['🔍', 'failure'],
  ['🏋', 'drill'],
  ['📅', 'volatile'],
  ['🧪', 'verify'],
  ['🎯', 'targeted'],
  ['↔', 'twice'],
];
const CALLOUT_RE = new RegExp(
  `^\\s*\\*\\*\\s*(${CALLOUTS.map(([marker]) => marker).join('|')})\\uFE0F?`,
  'u',
);

export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* ------------------------------------------------------------------ *
 * Link resolution
 * ------------------------------------------------------------------ */

function safeDecode(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function resolveLink(href, env) {
  if (!href) return href;
  if (/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(href)) return href; // absolute / mailto / protocol-relative
  if (href.startsWith('/')) return href;

  const hashAt = href.indexOf('#');
  const rawPath = hashAt === -1 ? href : href.slice(0, hashAt);
  const rawHash = hashAt === -1 ? '' : href.slice(hashAt + 1);
  const anchor = normaliseAnchor(safeDecode(rawHash));

  // A bare `#anchor` points inside the file we are currently rendering.
  const targetFile = rawPath === ''
    ? env.sourceFile
    : safeDecode(rawPath).replace(/^\.\//, '').replace(/^\.\.\//, '');

  const lookupKeys = [targetFile];
  if (!targetFile.includes('/')) lookupKeys.push(`ai-engineering-guide/${targetFile}`);

  for (const key of lookupKeys) {
    const entry = env.anchorMap?.get(key);
    if (!entry) continue;
    if (!anchor) return entry.default;
    const resolved = entry.anchors.get(anchor);
    if (!resolved) continue;
    // Staying on the same page? Keep it a pure fragment so the browser
    // scrolls instead of navigating.
    if (env.pageUrl && resolved.startsWith(`${env.pageUrl}#`)) {
      return resolved.slice(env.pageUrl.length);
    }
    if (env.pageUrl && resolved === env.pageUrl) return '#top';
    return resolved;
  }

  if (!rawPath && anchor) return `#${anchor}`;
  return href;
}

/* ------------------------------------------------------------------ *
 * Renderer
 * ------------------------------------------------------------------ */

export function createRenderer() {
  const md = new MarkdownIt({
    html: false,
    linkify: true,
    typographer: true,
    breaks: false,
  });

  // Only autolink things that actually declare a scheme. Without this,
  // `CONTRIBUTING.md` and `Q3-planning.md` in prose become links to a
  // nonexistent `.md` domain.
  md.linkify.set({ fuzzyLink: false, fuzzyEmail: false, fuzzyIP: false });

  // Fenced code: highlighted at build time, wrapped so the language label and
  // the copy button have somewhere to live.
  md.renderer.rules.fence = (tokens, index) => {
    const token = tokens[index];
    const requested = (token.info || '').trim().split(/\s+/)[0].toLowerCase();
    const language = LANGUAGE_ALIASES[requested] ?? requested;
    const known = language && hljs.getLanguage(language);

    let body;
    if (known) {
      try {
        body = hljs.highlight(token.content, { language, ignoreIllegals: true }).value;
      } catch {
        body = escapeHtml(token.content);
      }
    } else {
      body = escapeHtml(token.content);
    }

    const label = requested || 'text';
    const classes = `hljs${known ? ` language-${language}` : ''}`;
    return (
      `<div class="code-block" data-lang="${escapeHtml(label)}">` +
      `<pre><code class="${classes}">${body}</code></pre>` +
      `</div>\n`
    );
  };

  // Tables get their own scroll container — some of them are wide.
  md.renderer.rules.table_open = () => '<div class="table-wrap"><table>\n';
  md.renderer.rules.table_close = () => '</table></div>\n';

  // Links: rewrite md-to-md references, mark outbound ones.
  const defaultLink = md.renderer.rules.link_open
    ?? ((tokens, index, options, env, self) => self.renderToken(tokens, index, options));
  md.renderer.rules.link_open = (tokens, index, options, env, self) => {
    const token = tokens[index];
    const hrefIndex = token.attrIndex('href');
    if (hrefIndex >= 0) {
      const href = token.attrs[hrefIndex][1];
      const resolved = resolveLink(href, env);
      token.attrs[hrefIndex][1] = resolved;
      if (/^https?:/i.test(resolved)) {
        token.attrSet('rel', 'noopener noreferrer');
        token.attrSet('target', '_blank');
        token.attrJoin('class', 'external');
      }
    }
    return defaultLink(tokens, index, options, env, self);
  };

  // Headings: stable ids plus a quiet anchor affordance.
  md.renderer.rules.heading_open = (tokens, index, options, env, self) => {
    const token = tokens[index];
    const level = Number(token.tag.slice(1));
    const text = tokens[index + 1]?.content ?? '';

    // Question anchors were minted by the content parser; consume them in
    // document order so every link in the build agrees on the same id.
    let id;
    if (level === 3 && env.questionAnchors?.length) id = env.questionAnchors.shift();
    id = id ?? env.slugger(text);

    token.attrSet('id', id);
    token.attrJoin('class', 'anchored');
    env.headings?.push({ level, text, id });
    return self.renderToken(tokens, index, options);
  };
  md.renderer.rules.heading_close = (tokens, index, options, env, self) => {
    let id = null;
    for (let i = index - 1; i >= 0; i -= 1) {
      if (tokens[i].type === 'heading_open') {
        id = tokens[i].attrGet('id');
        break;
      }
    }
    const close = self.renderToken(tokens, index, options);
    return id
      ? `<a class="heading-link" href="#${id}" aria-hidden="true" tabindex="-1">#</a>${close}`
      : close;
  };

  // Callouts: a paragraph that opens with one of the guides' markers becomes
  // a boxed aside instead of running prose.
  md.renderer.rules.callout_open = (tokens, index) =>
    `<aside class="callout callout-${tokens[index].meta.kind}">\n`;
  md.renderer.rules.callout_close = () => '</aside>\n';

  /** Index of the token that closes the block opened at `start`. */
  function blockEnd(tokens, start) {
    const token = tokens[start];
    if (token.nesting !== 1) return start;
    const closeType = `${token.type.slice(0, -'_open'.length)}_close`;
    for (let k = start + 1; k < tokens.length; k += 1) {
      if (tokens[k].type === closeType && tokens[k].level === token.level) return k;
    }
    return start;
  }

  const CONTINUATION = new Set([
    'bullet_list_open',
    'ordered_list_open',
    'blockquote_open',
    'table_open',
    'fence',
    'code_block',
  ]);

  md.core.ruler.push('callouts', (state) => {
    const tokens = state.tokens;
    for (let i = 0; i < tokens.length; i += 1) {
      if (tokens[i].type !== 'paragraph_open') continue;
      const inline = tokens[i + 1];
      if (!inline || inline.type !== 'inline') continue;
      if (tokens[i + 2]?.type !== 'paragraph_close') continue;
      const match = CALLOUT_RE.exec(inline.content);
      if (!match) continue;

      const kind = CALLOUTS.find(([marker]) => marker === match[1])[1];

      // The inline stream can open with an empty text token, so find the
      // opening `<strong>` rather than assuming it is first.
      const strongOpen = (inline.children ?? []).find(
        (child) => child.type !== 'text' || child.content !== '',
      );
      if (strongOpen && strongOpen.type === 'strong_open') {
        strongOpen.attrJoin('class', 'callout-label');
      }

      // `💰 Math, step by step:` is followed by the steps. A trailing colon is
      // the guides' own signal that what comes next belongs to the callout.
      let end = i + 2;
      if (/:\**\s*$/.test(inline.content)) {
        const next = tokens[end + 1];
        if (next && next.level === tokens[i].level && CONTINUATION.has(next.type)) {
          end = blockEnd(tokens, end + 1);
        }
      }

      const open = new state.Token('callout_open', '', 1);
      open.meta = { kind };
      open.block = true;
      const close = new state.Token('callout_close', '', -1);
      close.block = true;

      tokens.splice(end + 1, 0, close);
      tokens.splice(i, 0, open);
      i = end + 2;
    }
  });

  return md;
}

/** Render markdown for one page, returning the HTML and the headings it contains. */
export function renderMarkdown(md, markdown, { sourceFile, anchorMap, pageUrl, questionAnchors } = {}) {
  const headings = [];
  const env = {
    sourceFile,
    anchorMap,
    pageUrl,
    headings,
    slugger: createSlugger(),
    questionAnchors: questionAnchors ? [...questionAnchors] : null,
  };
  const html = md.render(markdown, env);
  return { html, headings };
}
