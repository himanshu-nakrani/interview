// Page templates. Everything is plain string building — no framework, no
// hydration, no client-side routing. The HTML that ships is the HTML you read.

import { escapeHtml } from './markdown.mjs';
import { readingMinutes } from './content.mjs';

const SITE_NAME = 'Interview Guides';
const SITE_TAGLINE = 'Long-form interview preparation, set for reading.';

const FAVICON =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E" +
  "%3Crect width='32' height='32' rx='7' fill='%232f6f61'/%3E" +
  "%3Cpath d='M9 9.5h6.2c.9 0 1.8.5 1.8 1.4v12c0-.9-.9-1.4-1.8-1.4H9zM23 9.5h-6.2c-.9 0-1.8.5-1.8 1.4v12c0-.9.9-1.4 1.8-1.4H23z' fill='none' stroke='%23f7f3ea' stroke-width='1.6' stroke-linejoin='round'/%3E" +
  '%3C/svg%3E';

export const MARKERS = [
  ['⚠', 'trap', 'Trap', 'The misconception or silent failure that passes review and breaks later.'],
  ['🗣', 'say', 'Say this in the room', 'A short answer you can deliver more or less verbatim.'],
  ['📐', 'numbers', 'Numbers you must know', 'Memorisable constants, with their derivation.'],
  ['📄', 'paper', 'Paper', 'Author, year, contribution, and what it replaced.'],
  ['💰', 'math', 'Worked math', 'Cost or latency claims with the arithmetic shown.'],
  ['🔍', 'failure', 'Failure taxonomy', 'How the architecture breaks in production.'],
  ['🏋', 'drill', 'Drill', 'A timed, unaided exercise with pass criteria.'],
  ['📅', 'volatile', 'Volatile', 'Correct when written — re-verify before an interview loop.'],
  ['🧪', 'verify', 'Verify before you skip', 'Questions to answer before exercising a skip.'],
  ['🎯', 'targeted', 'Targeted', 'Relevant only to specific employer archetypes.'],
  ['↔', 'twice', 'Mechanism, then framework', 'Taught from scratch first, then through the framework.'],
];

/* ------------------------------------------------------------------ *
 * Small helpers
 * ------------------------------------------------------------------ */

const attr = (value) => escapeHtml(String(value ?? ''));

function formatNumber(value) {
  return Number(value).toLocaleString('en-US');
}

function pluralise(count, singular, plural = `${singular}s`) {
  return `${formatNumber(count)} ${count === 1 ? singular : plural}`;
}

const icon = {
  search:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4.5 4.5"/></svg>',
  type: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7V5h11v2M9.5 5v14M7 19h5"/><path d="M14 12v-1.2h6V12M17 10.8V19M15.5 19h3"/></svg>',
  contrast:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.2"/><path d="M12 3.8v16.4A8.2 8.2 0 0 0 12 3.8Z" fill="currentColor" stroke="none"/></svg>',
  menu: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>',
  close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg>',
  arrowLeft: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 12H5m0 0 6-6m-6 6 6 6"/></svg>',
  arrowRight: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14m0 0-6-6m6 6-6 6"/></svg>',
};

/* ------------------------------------------------------------------ *
 * Chrome
 * ------------------------------------------------------------------ */

function topbar({ collections, activeCollection }) {
  const links = collections
    .map(
      (collection) =>
        `<a class="topbar-link${collection.id === activeCollection ? ' is-current' : ''}" href="${attr(
          collection.base,
        )}">${escapeHtml(collection.short)}</a>`,
    )
    .join('');

  return `<header class="topbar">
  <div class="topbar-inner">
    <button class="icon-btn topbar-menu" type="button" data-action="toggle-nav" aria-label="Open contents" aria-expanded="false">${icon.menu}</button>
    <a class="brand" href="/"><span class="brand-mark" aria-hidden="true"></span><span class="brand-name">${escapeHtml(SITE_NAME)}</span></a>
    <nav class="topbar-nav" aria-label="Guides">${links}</nav>
    <div class="topbar-tools">
      <button class="search-trigger" type="button" data-action="open-search" aria-label="Search">
        ${icon.search}<span class="search-trigger-text">Search</span><kbd class="search-trigger-kbd">/</kbd>
      </button>
      <button class="icon-btn" type="button" data-action="toggle-prefs" aria-label="Reading settings" aria-expanded="false" aria-controls="prefs">${icon.type}</button>
      <button class="icon-btn" type="button" data-action="cycle-theme" aria-label="Change theme">${icon.contrast}</button>
    </div>
  </div>
  <div class="reading-progress" aria-hidden="true"><span></span></div>
</header>`;
}

function prefsPanel() {
  const group = (label, name, options) => `
    <div class="prefs-row">
      <span class="prefs-label" id="prefs-${name}">${escapeHtml(label)}</span>
      <div class="segmented" role="group" aria-labelledby="prefs-${name}" data-pref="${name}">
        ${options
          .map(
            (option) =>
              `<button type="button" value="${attr(option.value)}" title="${attr(option.title ?? option.label)}">${
                option.html ?? escapeHtml(option.label)
              }</button>`,
          )
          .join('')}
      </div>
    </div>`;

  return `<div class="prefs" id="prefs" hidden>
  <div class="prefs-inner">
    ${group('Theme', 'theme', [
      { value: 'system', label: 'Auto' },
      { value: 'light', label: 'Light' },
      { value: 'sepia', label: 'Sepia' },
      { value: 'dark', label: 'Dark' },
    ])}
    ${group('Type', 'font', [
      { value: 'serif', label: 'Serif', html: '<span style="font-family:var(--font-serif)">Serif</span>' },
      { value: 'sans', label: 'Sans', html: '<span style="font-family:var(--font-sans)">Sans</span>' },
    ])}
    ${group('Size', 'size', [
      { value: 'xs', label: 'A', title: 'Small', html: '<span style="font-size:.75em">A</span>' },
      { value: 's', label: 'A', title: 'Comfortable', html: '<span style="font-size:.9em">A</span>' },
      { value: 'm', label: 'A', title: 'Large', html: '<span style="font-size:1.05em">A</span>' },
      { value: 'l', label: 'A', title: 'Larger', html: '<span style="font-size:1.2em">A</span>' },
    ])}
    ${group('Width', 'width', [
      { value: 'narrow', label: 'Narrow' },
      { value: 'normal', label: 'Normal' },
      { value: 'wide', label: 'Wide' },
    ])}
    ${group('Spacing', 'leading', [
      { value: 'tight', label: 'Tight' },
      { value: 'normal', label: 'Normal' },
      { value: 'airy', label: 'Airy' },
    ])}
    <p class="prefs-note">Saved in this browser. <kbd>/</kbd> to search, <kbd>←</kbd> <kbd>→</kbd> to page.</p>
  </div>
</div>`;
}

function searchOverlay() {
  return `<div class="search-overlay" id="search" hidden>
  <div class="search-panel" role="dialog" aria-modal="true" aria-label="Search the guides">
    <div class="search-field">
      ${icon.search}
      <input type="search" id="search-input" placeholder="Search 5,600+ questions…" autocomplete="off" spellcheck="false" aria-label="Search">
      <button class="icon-btn" type="button" data-action="close-search" aria-label="Close search">${icon.close}</button>
    </div>
    <div class="search-results" id="search-results" role="listbox" aria-label="Results"></div>
    <div class="search-foot"><kbd>↑</kbd><kbd>↓</kbd> to move · <kbd>↵</kbd> to open · <kbd>esc</kbd> to close</div>
  </div>
</div>`;
}

function siteFooter() {
  return `<footer class="sitefoot">
  <div class="sitefoot-inner">
    <p>${escapeHtml(SITE_TAGLINE)} Built from the markdown in this repository — the files stay the source of truth.</p>
    <p class="sitefoot-meta"><a href="/">Home</a> · <a href="/ai-engineer/">AI Engineer</a> · <a href="/python-backend/">Python Backend</a> · <a href="/search/">Search</a></p>
  </div>
</footer>`;
}

/* ------------------------------------------------------------------ *
 * Sidebar
 * ------------------------------------------------------------------ */

function sidebarForCollection(collection, activeUrl) {
  const sectionLink = (section) =>
    `<li><a class="nav-section${section.url === activeUrl ? ' is-current' : ''}" href="${attr(section.url)}"${
      section.url === activeUrl ? ' aria-current="page"' : ''
    }><span class="nav-num">${section.number ?? ''}</span><span class="nav-text">${escapeHtml(
      section.title,
    )}</span></a></li>`;

  let body;
  if (collection.parts.length) {
    body = collection.parts
      .map((part) => {
        const open = part.url === activeUrl || part.sections.some((s) => s.url === activeUrl);
        return `<details class="nav-part"${open ? ' open' : ''}>
  <summary><span class="nav-part-label">${escapeHtml(part.label)}</span><span class="nav-part-title">${escapeHtml(
    part.title,
  )}</span></summary>
  <ul class="nav-list">${part.sections.map(sectionLink).join('')}</ul>
</details>`;
      })
      .join('');
  } else {
    body = `<ul class="nav-list nav-list-flat">${collection.sections.map(sectionLink).join('')}</ul>`;
  }

  const extras =
    collection.id === 'ai-engineer'
      ? `<ul class="nav-list nav-list-flat">
      <li><a class="nav-section${collection.base === activeUrl ? ' is-current' : ''}" href="${attr(
        collection.base,
      )}"><span class="nav-text">Overview</span></a></li>
      <li><a class="nav-section${collection.curriculum.url === activeUrl ? ' is-current' : ''}" href="${attr(
          collection.curriculum.url,
        )}"><span class="nav-text">Curriculum &amp; study paths</span></a></li>
    </ul>`
      : `<ul class="nav-list nav-list-flat">
      <li><a class="nav-section${collection.base === activeUrl ? ' is-current' : ''}" href="${attr(
        collection.base,
      )}"><span class="nav-text">Overview</span></a></li>
    </ul>`;

  return `<nav class="sidebar-nav" aria-label="${attr(collection.short)} contents">
  <p class="sidebar-title">${escapeHtml(collection.short)}</p>
  ${extras}
  ${body}
</nav>`;
}

/* ------------------------------------------------------------------ *
 * Outline (right rail)
 * ------------------------------------------------------------------ */

function outline(items, { label = 'On this page' } = {}) {
  if (!items || items.length < 2) return '';
  return `<nav class="outline-nav" aria-label="${attr(label)}">
  <p class="outline-title">${escapeHtml(label)}</p>
  <ol class="outline-list">
    ${items
      .map(
        (item) =>
          `<li class="outline-l${item.level}"><a href="#${attr(item.id)}">${escapeHtml(item.text)}</a></li>`,
      )
      .join('')}
  </ol>
</nav>`;
}

/* ------------------------------------------------------------------ *
 * Pager
 * ------------------------------------------------------------------ */

function pager(prev, next) {
  if (!prev && !next) return '';
  const link = (target, direction) =>
    target
      ? `<a class="pager-link pager-${direction}" href="${attr(target.url)}" rel="${direction}" data-nav="${direction}">
    <span class="pager-dir">${direction === 'prev' ? `${icon.arrowLeft} Previous` : `Next ${icon.arrowRight}`}</span>
    <span class="pager-title">${escapeHtml(target.label)}</span>
  </a>`
      : '<span class="pager-link pager-empty" aria-hidden="true"></span>';
  return `<nav class="pager" aria-label="Section navigation">${link(prev, 'prev')}${link(next, 'next')}</nav>`;
}

/* ------------------------------------------------------------------ *
 * Layout
 * ------------------------------------------------------------------ */

export function layout({
  title,
  description,
  url,
  bodyClass = '',
  collections,
  activeCollection = null,
  sidebar = '',
  outline: outlineHtml = '',
  main,
  breadcrumb = '',
}) {
  const pageTitle = url === '/' ? `${SITE_NAME} — ${SITE_TAGLINE}` : `${title} · ${SITE_NAME}`;
  const hasSidebar = Boolean(sidebar);
  const hasOutline = Boolean(outlineHtml);

  return `<!doctype html>
<html lang="en" class="no-js" data-url="${attr(url)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>${escapeHtml(pageTitle)}</title>
<meta name="description" content="${attr(description ?? SITE_TAGLINE)}">
<meta name="color-scheme" content="light dark">
<meta name="theme-color" content="#faf7f2" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#17161a" media="(prefers-color-scheme: dark)">
<meta property="og:type" content="article">
<meta property="og:site_name" content="${attr(SITE_NAME)}">
<meta property="og:title" content="${attr(title)}">
<meta property="og:description" content="${attr(description ?? SITE_TAGLINE)}">
<link rel="icon" href="${FAVICON}">
<link rel="stylesheet" href="/assets/reader.css">
<script>
(function(){var d=document.documentElement;d.classList.remove('no-js');try{var p=JSON.parse(localStorage.getItem('reader-prefs')||'{}');for(var k in p){if(p[k])d.setAttribute('data-'+k,p[k]);}}catch(e){}})();
</script>
</head>
<body class="${attr(bodyClass)}${hasSidebar ? ' has-sidebar' : ''}${hasOutline ? ' has-outline' : ''}">
<a class="skip-link" href="#content">Skip to the text</a>
${topbar({ collections, activeCollection })}
${prefsPanel()}
${searchOverlay()}
<div class="shell">
  ${hasSidebar ? `<aside class="sidebar" id="sidebar">${sidebar}</aside><div class="nav-scrim" data-action="toggle-nav" hidden></div>` : ''}
  <main class="main" id="content" tabindex="-1">
    ${breadcrumb}
    ${main}
  </main>
  ${hasOutline ? `<aside class="outline">${outlineHtml}</aside>` : ''}
</div>
<dialog class="qdialog" id="question-dialog">
  <div class="qdialog-head">
    <div class="qdialog-bar">
      <p class="qdialog-count" id="qdialog-count" aria-live="polite"></p>
      <button class="icon-btn" type="button" data-action="close-question" aria-label="Close question">${icon.close}</button>
    </div>
  </div>
  <div class="qdialog-body" id="qdialog-body" tabindex="-1"></div>
  <div class="qdialog-foot">
    <div class="qdialog-bar">
      <button class="btn" type="button" data-action="prev-question">${icon.arrowLeft} Previous</button>
      <button class="btn" type="button" data-action="next-question">Next ${icon.arrowRight}</button>
    </div>
  </div>
</dialog>
${siteFooter()}
<script src="/assets/reader.js" defer></script>
</body>
</html>`;
}

/* ------------------------------------------------------------------ *
 * Pages
 * ------------------------------------------------------------------ */

export function homePage({ collections, stats }) {
  const cards = collections
    .map((collection) => {
      const s = collection.stats;
      return `<a class="guide-card" href="${attr(collection.base)}">
    <span class="guide-card-eyebrow">${escapeHtml(collection.short)}</span>
    <h2 class="guide-card-title">${escapeHtml(collection.title)}</h2>
    <p class="guide-card-body">${escapeHtml(collection.tagline)}</p>
    <p class="guide-card-meta">${escapeHtml(
      [
        collection.parts.length ? pluralise(collection.parts.length, 'part') : null,
        pluralise(s.sections, 'section'),
        pluralise(s.questions, 'question'),
        `≈${formatNumber(Math.round(s.minutes / 60))} h of reading`,
      ]
        .filter(Boolean)
        .join(' · '),
    )}</p>
  </a>`;
    })
    .join('');

  const legend = MARKERS.map(
    ([marker, kind, label, blurb]) => `<li class="legend-item legend-${kind}">
    <span class="legend-marker" aria-hidden="true">${marker}</span>
    <span class="legend-body"><strong>${escapeHtml(label)}</strong> ${escapeHtml(blurb)}</span>
  </li>`,
  ).join('');

  const main = `<article class="reader home">
  <header class="hero">
    <p class="eyebrow">A quiet place to read</p>
    <h1 class="hero-title">Two interview guides,<br>written to be read end to end.</h1>
    <p class="hero-lede">${formatNumber(stats.questions)} questions and ${formatNumber(
      stats.words,
    )} words on AI engineering and Python backend engineering — set in a single reading column, with light, sepia and dark modes, and nothing that blinks.</p>
    <p class="hero-actions">
      <a class="btn btn-primary" href="/ai-engineer/">Start with AI Engineer</a>
      <a class="btn" href="/python-backend/">Python Backend</a>
    </p>
  </header>

  <section class="guide-cards">${cards}</section>

  <section class="legend-block">
    <h2 class="section-heading">The marks in the margin</h2>
    <p class="prose">Both guides use a small, consistent vocabulary of markers, set as quiet callouts
      so you can skim a section for exactly the thing you need.</p>
    <ul class="legend">${legend}</ul>
  </section>
</article>`;

  return layout({
    title: SITE_NAME,
    description: SITE_TAGLINE,
    url: '/',
    bodyClass: 'page-home',
    collections,
    main,
  });
}

export function collectionPage({ collection, collections, introHtml, headings }) {
  const s = collection.stats;
  const partsHtml = collection.parts.length
    ? `<section class="part-index">
  <h2 class="section-heading" id="contents">Contents</h2>
  <ol class="part-list">
    ${collection.parts
      .map(
        (part) => `<li class="part-item">
      <a class="part-link" href="${attr(part.url)}">
        <span class="part-label">${escapeHtml(part.label)}</span>
        <span class="part-title">${escapeHtml(part.title)}</span>
      </a>
      <ul class="part-sections">
        ${part.sections
          .map(
            (section) =>
              `<li><a href="${attr(section.url)}"><span class="sec-num">${section.number}</span>${escapeHtml(
                section.title,
              )}</a><span class="sec-meta">${formatNumber(section.questions.length)} q</span></li>`,
          )
          .join('')}
      </ul>
    </li>`,
      )
      .join('')}
  </ol>
</section>`
    : `<section class="part-index">
  <h2 class="section-heading" id="contents">Contents</h2>
  <ol class="chapter-list">
    ${collection.sections
      .map(
        (section) => `<li>
      <a href="${attr(section.url)}"><span class="sec-num">${section.number}</span><span class="sec-title">${escapeHtml(
        section.title,
      )}</span></a>
      <span class="sec-meta">${formatNumber(section.questions.length)} questions · ${section.minutes} min</span>
    </li>`,
      )
      .join('')}
  </ol>
</section>`;

  const first = collection.sections[0];
  const main = `<article class="reader">
  <header class="page-head">
    <p class="eyebrow">Guide</p>
    <h1 class="page-title">${escapeHtml(collection.title)}</h1>
    <p class="standfirst">${escapeHtml(collection.tagline)}</p>
    <p class="page-meta">${escapeHtml(
      [
        collection.parts.length ? pluralise(collection.parts.length, 'part') : null,
        pluralise(s.sections, 'section'),
        pluralise(s.questions, 'question'),
        `${formatNumber(s.words)} words`,
      ]
        .filter(Boolean)
        .join(' · '),
    )}</p>
    <p class="hero-actions">
      <a class="btn btn-primary" href="${attr(first.url)}">Start reading</a>
      ${
        collection.curriculum
          ? `<a class="btn" href="${attr(collection.curriculum.url)}">Curriculum &amp; study paths</a>`
          : ''
      }
    </p>
  </header>
  <div class="prose">${introHtml}</div>
  ${partsHtml}
</article>`;

  return layout({
    title: collection.title,
    description: collection.tagline,
    url: collection.base,
    bodyClass: 'page-collection',
    collections,
    activeCollection: collection.id,
    sidebar: sidebarForCollection(collection, collection.base),
    outline: outline(headings),
    main,
  });
}

export function partPage({ part, collection, collections, introHtml, prev, next }) {
  const questions = part.sections.reduce((total, section) => total + section.questions.length, 0);
  const words = part.sections.reduce((total, section) => total + section.words, 0);

  const main = `<article class="reader">
  <header class="page-head">
    <p class="eyebrow"><a href="${attr(collection.base)}">${escapeHtml(collection.short)}</a> · ${escapeHtml(
      part.label,
    )}</p>
    <h1 class="page-title">${escapeHtml(part.title)}</h1>
    <p class="page-meta">${escapeHtml(
      `${pluralise(part.sections.length, 'section')} · ${pluralise(questions, 'question')} · ${readingMinutes(
        words,
      )} min`,
    )}</p>
  </header>
  ${introHtml ? `<div class="prose standfirst-prose">${introHtml}</div>` : ''}
  <ol class="chapter-list">
    ${part.sections
      .map(
        (section) => `<li>
      <a href="${attr(section.url)}"><span class="sec-num">${section.number}</span><span class="sec-title">${escapeHtml(
        section.title,
      )}</span></a>
      ${section.kicker ? `<p class="sec-kicker">${escapeHtml(section.kicker)}</p>` : ''}
      <span class="sec-meta">${formatNumber(section.questions.length)} questions · ${section.minutes} min</span>
    </li>`,
      )
      .join('')}
  </ol>
  ${pager(prev, next)}
</article>`;

  return layout({
    title: `${part.label} — ${part.title}`,
    description: `${part.label} of the ${collection.title} guide: ${part.title}.`,
    url: part.url,
    bodyClass: 'page-part',
    collections,
    activeCollection: collection.id,
    sidebar: sidebarForCollection(collection, part.url),
    main,
  });
}

export function sectionPage({ section, collection, collections, bodyHtml, prev, next }) {
  const crumbs = [
    `<a href="${attr(collection.base)}">${escapeHtml(collection.short)}</a>`,
    section.part ? `<a href="${attr(section.part.url)}">${escapeHtml(section.part.label)}</a>` : null,
    section.number != null ? `<span>Section ${section.number}</span>` : null,
  ]
    .filter(Boolean)
    .join('<span class="crumb-sep" aria-hidden="true">·</span>');

  const main = `<article class="reader">
  <header class="page-head">
    <p class="eyebrow">${crumbs}</p>
    <h1 class="page-title">${escapeHtml(section.title)}</h1>
    ${section.kicker ? `<p class="standfirst">${escapeHtml(section.kicker)}</p>` : ''}
    <p class="page-meta">${escapeHtml(
      `${pluralise(section.questions.length, 'question')} · ${section.minutes} min · ${formatNumber(
        section.words,
      )} words`,
    )}</p>
  </header>
  <div class="prose">${bodyHtml}</div>
  ${pager(prev, next)}
</article>`;

  const questionOutline = section.questions.map((question) => ({
    level: 3,
    id: question.anchor,
    text: question.title,
  }));

  return layout({
    title: section.fullTitle,
    description: section.kicker ?? `${section.fullTitle} — ${collection.title} interview guide.`,
    url: section.url,
    bodyClass: 'page-section',
    collections,
    activeCollection: collection.id,
    sidebar: sidebarForCollection(collection, section.url),
    outline: outline(questionOutline, { label: 'Questions' }),
    main,
  });
}

export function articlePage({ title, standfirst, url, collection, collections, bodyHtml, headings, meta }) {
  const main = `<article class="reader">
  <header class="page-head">
    <p class="eyebrow"><a href="${attr(collection.base)}">${escapeHtml(collection.short)}</a></p>
    <h1 class="page-title">${escapeHtml(title)}</h1>
    ${standfirst ? `<p class="standfirst">${escapeHtml(standfirst)}</p>` : ''}
    ${meta ? `<p class="page-meta">${escapeHtml(meta)}</p>` : ''}
  </header>
  <div class="prose">${bodyHtml}</div>
</article>`;

  return layout({
    title,
    description: standfirst ?? title,
    url,
    bodyClass: 'page-article',
    collections,
    activeCollection: collection.id,
    sidebar: sidebarForCollection(collection, url),
    // The curriculum lists all 87 sections as h3s; the rail only wants the parts.
    outline: outline((headings ?? []).filter((heading) => heading.level <= 2)),
    main,
  });
}

export function searchPage({ collections }) {
  const main = `<article class="reader">
  <header class="page-head">
    <p class="eyebrow">Search</p>
    <h1 class="page-title">Find a question</h1>
    <p class="standfirst">Every section and every question across both guides. Type a few words — “kv cache”,
      “dpo”, “asyncio cancel”, “offer negotiation”.</p>
  </header>
  <div class="search-page">
    <div class="search-field search-field-inline">
      ${icon.search}
      <input type="search" id="search-page-input" placeholder="Search the guides…" autocomplete="off" spellcheck="false" aria-label="Search the guides">
    </div>
    <div class="search-results" id="search-page-results"></div>
    <noscript><p class="prose">Search needs JavaScript. Without it, use the contents lists:
      ${collections.map((c) => `<a href="${attr(c.base)}">${escapeHtml(c.short)}</a>`).join(' · ')}.</p></noscript>
  </div>
</article>`;

  return layout({
    title: 'Search',
    description: 'Search every question across both interview guides.',
    url: '/search/',
    bodyClass: 'page-search',
    collections,
    main,
  });
}

export function notFoundPage({ collections }) {
  const main = `<article class="reader">
  <header class="page-head">
    <p class="eyebrow">404</p>
    <h1 class="page-title">That page isn’t here.</h1>
    <p class="standfirst">The link may be stale, or the section may have moved. Both tables of contents are
      one click away.</p>
    <p class="hero-actions">
      <a class="btn btn-primary" href="/">Home</a>
      <a class="btn" href="/search/">Search</a>
    </p>
  </header>
</article>`;

  return layout({
    title: 'Not found',
    description: 'Page not found.',
    url: '/404.html',
    bodyClass: 'page-404',
    collections,
    main,
  });
}
