// Reads the markdown sources and turns them into a page tree.
//
// The markdown files are the single source of truth and are never modified.
// Everything the site knows — parts, sections, questions, anchors, counts —
// is derived here.

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { createSlugger, normaliseAnchor, slugify } from './slug.mjs';

const WORDS_PER_MINUTE = 220;

/* ------------------------------------------------------------------ *
 * Line scanning
 * ------------------------------------------------------------------ */

/**
 * Walk lines, reporting which ones are real ATX headings. Lines inside fenced
 * code blocks or indented code look like headings often enough in this corpus
 * (`# dis output for ...`, `## 2. Built-in ...` in a docstring) that ignoring
 * fences would shred the document.
 */
function* scanLines(markdown) {
  const lines = markdown.split('\n');
  let fence = null;

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const fenceMatch = /^ {0,3}(`{3,}|~{3,})/.exec(line);

    if (fence) {
      if (fenceMatch && fenceMatch[1][0] === fence[0] && fenceMatch[1].length >= fence.length) {
        fence = null;
      }
      yield { index: i, line, heading: null };
      continue;
    }

    if (fenceMatch) {
      fence = fenceMatch[1];
      yield { index: i, line, heading: null };
      continue;
    }

    const heading = /^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
    yield {
      index: i,
      line,
      heading: heading ? { level: heading[1].length, text: heading[2].trim() } : null,
    };
  }
}

/** Split a document into `{ heading, body }` blocks at the given heading level. */
export function splitAtLevel(markdown, level) {
  const blocks = [];
  let preamble = [];
  let current = null;

  for (const { line, heading } of scanLines(markdown)) {
    if (heading && heading.level === level) {
      if (current) blocks.push(current);
      current = { title: heading.text, lines: [] };
      continue;
    }
    if (current) current.lines.push(line);
    else preamble.push(line);
  }
  if (current) blocks.push(current);

  return {
    preamble: trimBlock(preamble),
    blocks: blocks.map((block) => ({ title: block.title, body: trimBlock(block.lines) })),
  };
}

/** Drop leading/trailing blank lines and the `---` rules that separate sections. */
function trimBlock(lines) {
  const out = [...lines];
  const isNoise = (line) => line.trim() === '' || /^ {0,3}(-{3,}|\*{3,}|_{3,})\s*$/.test(line);
  while (out.length && isNoise(out[0])) out.shift();
  while (out.length && isNoise(out[out.length - 1])) out.pop();
  return out.join('\n');
}

/** Headings of a given level inside a body, in document order. */
function headingsAtLevel(markdown, level) {
  const found = [];
  for (const { heading } of scanLines(markdown)) {
    if (heading && heading.level === level) found.push(heading.text);
  }
  return found;
}

function countWords(markdown) {
  const prose = markdown
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`]*`/g, ' ');
  const matches = prose.match(/[\p{L}\p{N}][\p{L}\p{N}'’-]*/gu);
  return matches ? matches.length : 0;
}

export function readingMinutes(words) {
  return Math.max(1, Math.round(words / WORDS_PER_MINUTE));
}

/* ------------------------------------------------------------------ *
 * URL helpers
 * ------------------------------------------------------------------ */

// Words that read as noise when a URL is truncated mid-title.
const TRAILING_NOISE = new Set([
  'a', 'an', 'and', 'as', 'at', 'by', 'for', 'from', 'in', 'is', 'of', 'on',
  'or', 'the', 'to', 'vs', 'with',
]);

/** Short, readable path segment: `07-math-for-llm-engineers`. */
function pathSegment(number, title, taken, { maxLength = 46 } = {}) {
  const prefix = number == null ? '' : `${String(number).padStart(2, '0')}-`;
  const words = slugify(title).split('-').filter(Boolean);
  let kept = [];
  let length = 0;
  for (const word of words) {
    if (/^\d+$/.test(word) && kept.length === 0) continue; // the number is already the prefix
    const next = length === 0 ? word.length : length + 1 + word.length;
    if (next > maxLength) break;
    kept.push(word);
    length = next;
  }
  while (kept.length > 1 && TRAILING_NOISE.has(kept[kept.length - 1])) kept.pop();
  const tail = kept.join('-');
  let candidate = `${prefix}${tail}`.replace(/-$/, '') || `${prefix}section`.replace(/-$/, '');
  let suffix = 2;
  while (taken.has(candidate)) candidate = `${prefix}${tail}-${suffix++}`;
  taken.add(candidate);
  return candidate;
}

/* ------------------------------------------------------------------ *
 * Parsing
 * ------------------------------------------------------------------ */

function parseSection({ title, body, sourceFile, collection, taken }) {
  // `## 7. Math for LLM Engineers: …` — the leading number is the global
  // section number used throughout the guides ("see §34").
  const numbered = /^(\d+)\.\s+(.*)$/.exec(title);
  const number = numbered ? Number(numbered[1]) : null;
  const heading = numbered ? numbered[2].trim() : title;

  // Many sections open with an italic one-liner explaining what mastering the
  // section proves. It reads better as a standfirst than as body copy.
  let kicker = null;
  let content = body;
  const kickerMatch = /^\*([^*][\s\S]*?)\*\s*(?:\n|$)/.exec(body);
  if (kickerMatch && !kickerMatch[1].includes('\n\n')) {
    kicker = kickerMatch[1].replace(/\s+/g, ' ').trim();
    content = body.slice(kickerMatch[0].length).replace(/^\s+/, '');
  }

  const questionTitles = headingsAtLevel(content, 3);
  const slugger = createSlugger();
  const questions = questionTitles.map((text) => ({ title: text, anchor: slugger(text) }));

  const segment = pathSegment(number, heading, taken);
  const url = `${collection.base}${segment}/`;
  const words = countWords(body);

  return {
    kind: 'section',
    collection: collection.id,
    sourceFile,
    number,
    title: heading,
    fullTitle: number == null ? heading : `${number}. ${heading}`,
    anchor: slugify(title),
    kicker,
    markdown: content,
    questions,
    words,
    minutes: readingMinutes(words),
    url,
  };
}

function parsePartFile(path, sourceFile, collection, taken) {
  const raw = readFileSync(path, 'utf8');
  const { preamble: docPreamble, blocks: h1Blocks } = splitAtLevel(raw, 1);
  const doc = h1Blocks[0] ?? { title: sourceFile, body: docPreamble };

  const rawTitle = doc.title;
  // `PART IV — Inference, Serving and AI Infrastructure`
  const partMatch = /^PART\s+([0-9IVXL]+)\s*[—–-]\s*(.+)$/i.exec(rawTitle);
  const label = partMatch ? `Part ${partMatch[1]}` : null;
  const title = partMatch ? partMatch[2].trim() : rawTitle;

  const { preamble, blocks } = splitAtLevel(doc.body, 2);
  const sections = blocks
    .filter((block) => !/^contents$/i.test(block.title))
    .map((block) => parseSection({ ...block, sourceFile, collection, taken }));

  const fileNumber = /part-(\d+)/.exec(sourceFile);
  const number = fileNumber ? Number(fileNumber[1]) : null;
  const segment = pathSegment(null, `part ${String(number ?? 0).padStart(2, '0')} ${title}`, taken, {
    maxLength: 40,
  });

  return {
    kind: 'part',
    collection: collection.id,
    sourceFile,
    number,
    label: label ?? `Part ${number ?? ''}`.trim(),
    title,
    intro: preamble,
    sections,
    url: `${collection.base}${segment}/`,
  };
}

function parseFlatFile(path, sourceFile, collection, taken) {
  const raw = readFileSync(path, 'utf8');
  const { blocks: h1Blocks, preamble: docPreamble } = splitAtLevel(raw, 1);
  const doc = h1Blocks[0] ?? { title: collection.title, body: docPreamble };
  const { preamble, blocks } = splitAtLevel(doc.body, 2);

  const sections = blocks
    .filter((block) => !/^contents$/i.test(block.title))
    .map((block) => parseSection({ ...block, sourceFile, collection, taken }));

  return { title: doc.title, intro: preamble, sections };
}

/* ------------------------------------------------------------------ *
 * Anchor map
 * ------------------------------------------------------------------ */

function registerAnchors(map, sourceFile, entries, defaultUrl) {
  const byFile = map.get(sourceFile) ?? { default: defaultUrl, anchors: new Map() };
  byFile.default = byFile.default ?? defaultUrl;
  for (const [anchor, url] of entries) {
    const key = normaliseAnchor(anchor);
    if (key && !byFile.anchors.has(key)) byFile.anchors.set(key, url);
  }
  map.set(sourceFile, byFile);
}

/* ------------------------------------------------------------------ *
 * Entry point
 * ------------------------------------------------------------------ */

export function loadContent(root) {
  const anchorMap = new Map();

  /* ---- Collection 1: the AI engineering guide ---- */
  const aiBase = '/ai-engineer/';
  const aiCollection = {
    id: 'ai-engineer',
    base: aiBase,
    title: 'AI Engineer (Generative AI)',
    short: 'AI Engineer',
    tagline: 'Interview preparation for AI Engineer, Applied AI and GenAI roles.',
  };
  const aiTaken = new Set(['curriculum']);

  const guideDir = join(root, 'ai-engineering-guide');
  const partFiles = readdirSync(guideDir)
    .filter((name) => name.endsWith('.md'))
    .sort();

  const parts = partFiles.map((name) =>
    parsePartFile(join(guideDir, name), `ai-engineering-guide/${name}`, aiCollection, aiTaken),
  );
  parts.sort((a, b) => (a.number ?? 0) - (b.number ?? 0));

  const aiSections = parts.flatMap((part) => {
    for (const section of part.sections) section.part = part;
    return part.sections;
  });
  aiCollection.parts = parts;
  aiCollection.sections = aiSections;

  /* ---- Collection 2: the Python backend guide ---- */
  const pyBase = '/python-backend/';
  const pyCollection = {
    id: 'python-backend',
    base: pyBase,
    title: 'Python Backend Engineering',
    short: 'Python Backend',
    tagline: 'Interview preparation for senior Python backend roles.',
  };
  const pyTaken = new Set();
  const pyDoc = parseFlatFile(
    join(root, 'python_backend_interview_prep.md'),
    'python_backend_interview_prep.md',
    pyCollection,
    pyTaken,
  );
  pyCollection.parts = [];
  pyCollection.sections = pyDoc.sections;
  pyCollection.intro = pyDoc.intro;
  pyCollection.documentTitle = pyDoc.title;

  /* ---- Standalone pages ---- */
  const readmeRaw = readFileSync(join(root, 'README.md'), 'utf8');
  const readme = splitAtLevel(readmeRaw, 1);
  const curriculumRaw = readFileSync(join(root, 'CURRICULUM.md'), 'utf8');
  const curriculum = splitAtLevel(curriculumRaw, 1);

  const aiIntro = {
    title: readme.blocks[0]?.title ?? aiCollection.title,
    markdown: readme.blocks[0]?.body ?? readme.preamble,
  };
  const curriculumPage = {
    title: curriculum.blocks[0]?.title ?? 'Curriculum',
    markdown: curriculum.blocks[0]?.body ?? curriculum.preamble,
    url: `${aiBase}curriculum/`,
    sourceFile: 'CURRICULUM.md',
  };
  aiCollection.intro = aiIntro;
  aiCollection.curriculum = curriculumPage;

  /* ---- Anchors, so cross-file markdown links keep working ---- */
  registerAnchors(anchorMap, 'README.md', [], aiBase);
  registerAnchors(
    anchorMap,
    'CURRICULUM.md',
    headingsAtLevel(curriculumPage.markdown, 2)
      .concat(headingsAtLevel(curriculumPage.markdown, 3))
      .map((text) => [slugify(text), `${curriculumPage.url}#${slugify(text)}`]),
    curriculumPage.url,
  );

  for (const collection of [aiCollection, pyCollection]) {
    const bySource = new Map();
    for (const section of collection.sections) {
      const entries = bySource.get(section.sourceFile) ?? [];
      entries.push([section.anchor, section.url]);
      for (const question of section.questions) {
        entries.push([question.anchor, `${section.url}#${question.anchor}`]);
      }
      bySource.set(section.sourceFile, entries);
    }
    for (const [sourceFile, entries] of bySource) {
      const part = collection.parts.find((candidate) => candidate.sourceFile === sourceFile);
      registerAnchors(anchorMap, sourceFile, entries, part ? part.url : collection.base);
    }
  }

  return { collections: [aiCollection, pyCollection], anchorMap };
}
