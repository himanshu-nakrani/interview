#!/usr/bin/env node
// Builds the static site into ./site — one HTML page per section, plus the
// index pages, the search index and the assets. The markdown sources are only
// ever read.

import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { loadContent, readingMinutes, splitAtLevel } from './lib/content.mjs';
import { createRenderer, renderMarkdown } from './lib/markdown.mjs';
import {
  articlePage,
  collectionPage,
  homePage,
  notFoundPage,
  partPage,
  searchPage,
  sectionPage,
} from './lib/templates.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const outDir = join(root, 'site');
const siteUrl = (process.env.SITE_URL || '').replace(/\/+$/, '');
// Where the site is rooted once deployed — only 404.html needs it, since the
// server hands that page out for URLs at any depth.
const basePath = `/${(process.env.BASE_PATH || '/').replace(/^\/+|\/+$/g, '')}/`.replace('//', '/');

/* README blocks the site renders better itself, or that are unusable as-is. */
const README_SKIP = new Set([
  'read it on the web', // build instructions — you are already here
  'parts',              // regenerated below with live section counts
  'all 87 sections',    // regenerated below
  'study paths',        // the table in README.md is truncated; CURRICULUM.md has it
]);

const started = Date.now();
const md = createRenderer();
const { collections, anchorMap } = loadContent(root);
const [ai, py] = collections;

/* ------------------------------------------------------------------ *
 * Stats
 * ------------------------------------------------------------------ */

for (const collection of collections) {
  const words = collection.sections.reduce((total, section) => total + section.words, 0);
  collection.stats = {
    sections: collection.sections.length,
    questions: collection.sections.reduce((total, section) => total + section.questions.length, 0),
    words,
    minutes: readingMinutes(words),
  };
}

const stats = {
  sections: collections.reduce((total, c) => total + c.stats.sections, 0),
  questions: collections.reduce((total, c) => total + c.stats.questions, 0),
  words: collections.reduce((total, c) => total + c.stats.words, 0),
  minutes: collections.reduce((total, c) => total + c.stats.minutes, 0),
};

/* ------------------------------------------------------------------ *
 * Render
 * ------------------------------------------------------------------ */

const pages = [];
const unresolved = [];

function render(markdown, options) {
  const result = renderMarkdown(md, markdown, { anchorMap, ...options });
  for (const match of result.html.matchAll(/href="([^"]*\.md(?:#[^"]*)?)"/g)) {
    unresolved.push({ page: options.pageUrl, href: match[1] });
  }
  return result;
}

function add(url, html) {
  pages.push({ url, html });
}

/* Home */
add('/', homePage({ collections, stats }));

/* AI guide overview — README prose, minus the blocks the site regenerates. */
{
  const { preamble, blocks } = splitAtLevel(ai.intro.markdown, 2);
  const kept = blocks.filter((block) => !README_SKIP.has(block.title.trim().toLowerCase()));
  const markdown = [preamble, ...kept.map((block) => `## ${block.title}\n\n${block.body}`)]
    .filter(Boolean)
    .join('\n\n');
  const { html, headings } = render(markdown, { sourceFile: 'README.md', pageUrl: ai.base });
  add(ai.base, collectionPage({ collection: ai, collections, introHtml: html, headings }));
}

/* Curriculum */
{
  const { html, headings } = render(ai.curriculum.markdown, {
    sourceFile: ai.curriculum.sourceFile,
    pageUrl: ai.curriculum.url,
  });
  add(
    ai.curriculum.url,
    articlePage({
      title: 'Curriculum',
      standfirst:
        'Why the guide is shaped the way it is: the answer-format contract, the recurring devices, the study paths, and the section-by-section plan.',
      url: ai.curriculum.url,
      collection: ai,
      collections,
      bodyHtml: html,
      headings,
      meta: `${ai.parts.length} parts · ${ai.stats.sections} sections`,
    }),
  );
}

/* Parts */
ai.parts.forEach((part, index) => {
  const { html } = render(part.intro ?? '', { sourceFile: part.sourceFile, pageUrl: part.url });
  const prevPart = ai.parts[index - 1];
  const nextPart = ai.parts[index + 1];
  add(
    part.url,
    partPage({
      part,
      collection: ai,
      collections,
      introHtml: html,
      prev: prevPart ? { url: prevPart.url, label: `${prevPart.label} — ${prevPart.title}` } : null,
      next: nextPart ? { url: nextPart.url, label: `${nextPart.label} — ${nextPart.title}` } : null,
    }),
  );
});

/* Python guide overview */
{
  const { html, headings } = render(py.intro ?? '', {
    sourceFile: 'python_backend_interview_prep.md',
    pageUrl: py.base,
  });
  add(py.base, collectionPage({ collection: py, collections, introHtml: html, headings }));
}

/* Sections */
for (const collection of collections) {
  collection.sections.forEach((section, index) => {
    const { html } = render(section.markdown, {
      sourceFile: section.sourceFile,
      pageUrl: section.url,
      questionAnchors: section.questions.map((question) => question.anchor),
    });
    const prev = collection.sections[index - 1];
    const next = collection.sections[index + 1];
    add(
      section.url,
      sectionPage({
        section,
        collection,
        collections,
        bodyHtml: html,
        prev: prev ? { url: prev.url, label: prev.fullTitle } : null,
        next: next ? { url: next.url, label: next.fullTitle } : null,
      }),
    );
  });
}

/* Utility pages */
add('/search/', searchPage({ collections }));
add('/404.html', notFoundPage({ collections }));

/* ------------------------------------------------------------------ *
 * Search index
 * ------------------------------------------------------------------ */

const indexPages = [];
const indexItems = [];
for (const collection of collections) {
  for (const section of collection.sections) {
    const pageIndex = indexPages.length;
    indexPages.push([
      section.url,
      section.fullTitle,
      collection.short,
      section.part ? section.part.label : '',
    ]);
    indexItems.push([pageIndex, '', '']);
    for (const question of section.questions) {
      indexItems.push([pageIndex, question.anchor, question.title]);
    }
  }
}
const searchIndex = { pages: indexPages, items: indexItems };

/* ------------------------------------------------------------------ *
 * Write
 * ------------------------------------------------------------------ */

/** Rewrite absolute site paths to relative ones so the build works at any base. */
function localise(html, pageUrl) {
  const dir = pageUrl.replace(/[^/]*$/, '');
  const depth = dir.split('/').filter(Boolean).length;
  // 404.html is served for URLs at any depth, so relative links would break.
  // It is the one page that needs to know where the site is rooted.
  const prefix = pageUrl === '/404.html' ? basePath : depth === 0 ? './' : '../'.repeat(depth);
  return html
    .replace(/((?:href|src)=")\/([^"]*)"/g, (_, attribute, rest) => `${attribute}${prefix}${rest}"`)
    .replace('<html ', `<html data-base="${prefix}" `);
}

function outputPath(url) {
  if (url.endsWith('.html')) return join(outDir, url.slice(1));
  return join(outDir, url.slice(1), 'index.html');
}

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

const known = new Set(pages.map((page) => page.url));
const broken = [];

for (const page of pages) {
  for (const match of page.html.matchAll(/href="(\/[^"#]*)(#[^"]*)?"/g)) {
    const target = match[1];
    if (target.startsWith('/assets/')) continue;
    if (!known.has(target)) broken.push({ page: page.url, target });
  }
  const file = outputPath(page.url);
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, localise(page.html, page.url));
}

mkdirSync(join(outDir, 'assets'), { recursive: true });
cpSync(join(root, 'web', 'reader.css'), join(outDir, 'assets', 'reader.css'));
cpSync(join(root, 'web', 'reader.js'), join(outDir, 'assets', 'reader.js'));
writeFileSync(join(outDir, 'assets', 'search-index.json'), JSON.stringify(searchIndex));
writeFileSync(join(outDir, '.nojekyll'), '');

if (siteUrl) {
  const urls = pages
    .filter((page) => !page.url.endsWith('.html'))
    .map((page) => `  <url><loc>${siteUrl}${page.url}</loc></url>`)
    .join('\n');
  writeFileSync(
    join(outDir, 'sitemap.xml'),
    `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`,
  );
  writeFileSync(join(outDir, 'robots.txt'), `User-agent: *\nAllow: /\nSitemap: ${siteUrl}/sitemap.xml\n`);
} else {
  writeFileSync(join(outDir, 'robots.txt'), 'User-agent: *\nAllow: /\n');
}

/* ------------------------------------------------------------------ *
 * Report
 * ------------------------------------------------------------------ */

const bytes = pages.reduce((total, page) => total + Buffer.byteLength(page.html), 0);
console.log(
  `built ${pages.length} pages · ${stats.sections} sections · ${stats.questions.toLocaleString('en-US')} questions · ` +
    `${(bytes / 1024 / 1024).toFixed(1)} MB html · ${((Date.now() - started) / 1000).toFixed(1)}s`,
);

if (unresolved.length) {
  const sample = unresolved.slice(0, 12);
  console.warn(`\n${unresolved.length} markdown link(s) could not be resolved to a page:`);
  for (const item of sample) console.warn(`  ${item.page} → ${item.href}`);
  if (unresolved.length > sample.length) console.warn(`  … and ${unresolved.length - sample.length} more`);
}

if (broken.length) {
  const seen = new Map();
  for (const item of broken) seen.set(`${item.page} → ${item.target}`, item);
  console.warn(`\n${seen.size} internal link(s) point at a page that was not generated:`);
  for (const key of [...seen.keys()].slice(0, 12)) console.warn(`  ${key}`);
  process.exitCode = 1;
}
