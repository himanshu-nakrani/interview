#!/usr/bin/env node
// A tiny static server for ./site — enough to read the built site locally.
// No dependencies, no watching, no reload. `npm run build` then `npm run serve`.

import { createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve } from 'node:path';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'site');
const port = Number(process.env.PORT || 4173);

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.xml': 'application/xml; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
};

function resolveFile(pathname) {
  const clean = normalize(decodeURIComponent(pathname)).replace(/^(\.\.[/\\])+/, '');
  const candidates = clean.endsWith('/')
    ? [join(root, clean, 'index.html')]
    : [join(root, clean), join(root, clean, 'index.html')];
  for (const candidate of candidates) {
    if (!candidate.startsWith(root)) continue;
    try {
      if (statSync(candidate).isFile()) return candidate;
    } catch {
      /* keep looking */
    }
  }
  return null;
}

createServer((request, response) => {
  const { pathname } = new URL(request.url, 'http://localhost');
  const file = resolveFile(pathname) ?? join(root, '404.html');
  let status = 200;
  try {
    statSync(file);
    if (file.endsWith('404.html') && !pathname.endsWith('404.html')) status = 404;
  } catch {
    response.writeHead(404, { 'content-type': 'text/plain' }).end('Not found');
    return;
  }
  response.writeHead(status, {
    'content-type': TYPES[extname(file)] ?? 'application/octet-stream',
    'cache-control': 'no-cache',
  });
  createReadStream(file).pipe(response);
}).listen(port, () => {
  console.log(`reading at http://localhost:${port}/`);
});
