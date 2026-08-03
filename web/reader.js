/* Interview Guides — progressive enhancement only.
 * The page is complete without this file; everything here is comfort. */

(function () {
  'use strict';

  var root = document.documentElement;
  var base = root.getAttribute('data-base') || './';
  var body = document.body;

  function $(selector, scope) { return (scope || document).querySelector(selector); }
  function $$(selector, scope) { return Array.prototype.slice.call((scope || document).querySelectorAll(selector)); }
  function href(path) { return base + String(path).replace(/^\//, ''); }
  function isTyping(el) {
    return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable);
  }

  /* --- Reading preferences ---------------------------------------- */

  var PREF_KEY = 'reader-prefs';
  var DEFAULTS = { theme: 'system', font: 'serif', size: 's', width: 'normal', leading: 'normal' };
  var THEME_ORDER = ['system', 'light', 'sepia', 'dark'];
  var prefs = {};

  try { prefs = JSON.parse(localStorage.getItem(PREF_KEY) || '{}') || {}; } catch (e) { prefs = {}; }

  function pref(name) { return prefs[name] || DEFAULTS[name]; }

  function applyPrefs() {
    Object.keys(DEFAULTS).forEach(function (name) {
      root.setAttribute('data-' + name, pref(name));
    });
    $$('.segmented').forEach(function (group) {
      var name = group.getAttribute('data-pref');
      $$('button', group).forEach(function (button) {
        button.setAttribute('aria-pressed', String(button.value === pref(name)));
      });
    });
  }

  function setPref(name, value) {
    prefs[name] = value;
    try { localStorage.setItem(PREF_KEY, JSON.stringify(prefs)); } catch (e) { /* private mode */ }
    applyPrefs();
  }

  applyPrefs();

  $$('.segmented').forEach(function (group) {
    group.addEventListener('click', function (event) {
      var button = event.target.closest('button');
      if (button) setPref(group.getAttribute('data-pref'), button.value);
    });
  });

  /* --- Panels ------------------------------------------------------ */

  var prefsPanel = $('#prefs');
  var prefsButton = $('[data-action="toggle-prefs"]');

  function setPrefsOpen(open) {
    if (!prefsPanel) return;
    prefsPanel.hidden = !open;
    if (prefsButton) prefsButton.setAttribute('aria-expanded', String(open));
  }

  function setNavOpen(open) {
    body.classList.toggle('nav-open', open);
    var button = $('[data-action="toggle-nav"]');
    if (button) button.setAttribute('aria-expanded', String(open));
    var scrim = $('.nav-scrim');
    if (scrim) scrim.hidden = !open;
  }

  document.addEventListener('click', function (event) {
    var trigger = event.target.closest('[data-action]');
    var action = trigger && trigger.getAttribute('data-action');

    if (action === 'toggle-prefs') { setPrefsOpen(prefsPanel.hidden); return; }
    if (action === 'toggle-nav') { setNavOpen(!body.classList.contains('nav-open')); return; }
    if (action === 'cycle-theme') {
      var next = THEME_ORDER[(THEME_ORDER.indexOf(pref('theme')) + 1) % THEME_ORDER.length];
      setPref('theme', next);
      setPrefsOpen(true);
      return;
    }
    if (action === 'open-search') { event.preventDefault(); openSearch(); return; }
    if (action === 'close-search') { closeSearch(); return; }

    if (prefsPanel && !prefsPanel.hidden && !event.target.closest('#prefs') && !event.target.closest('[data-action="toggle-prefs"]')) {
      setPrefsOpen(false);
    }
  });

  /* --- Copy buttons ------------------------------------------------ */

  $$('.code-block').forEach(function (block) {
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'code-copy';
    button.textContent = 'Copy';
    button.addEventListener('click', function () {
      var code = $('code', block);
      var text = code ? code.textContent : '';
      var done = function () {
        button.textContent = 'Copied';
        button.setAttribute('data-copied', '');
        setTimeout(function () {
          button.textContent = 'Copy';
          button.removeAttribute('data-copied');
        }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      } else {
        var area = document.createElement('textarea');
        area.value = text;
        document.body.appendChild(area);
        area.select();
        try { document.execCommand('copy'); done(); } catch (e) { /* ignore */ }
        document.body.removeChild(area);
      }
    });
    block.appendChild(button);
  });

  /* --- Reading progress -------------------------------------------- */

  var progress = $('.reading-progress span');
  var article = $('.main .reader');

  /* --- Outline scroll-spy ------------------------------------------ */

  var outlineLinks = $$('.outline-list a');
  var spyTargets = [];

  function measure() {
    spyTargets = outlineLinks
      .map(function (link) {
        var id = decodeURIComponent(link.getAttribute('href').slice(1));
        var target = document.getElementById(id);
        return target ? { link: link, top: target.getBoundingClientRect().top + window.scrollY } : null;
      })
      .filter(Boolean)
      .sort(function (a, b) { return a.top - b.top; });
  }

  var active = null;
  function onScroll() {
    if (progress && article) {
      var start = article.offsetTop;
      var span = article.offsetHeight - window.innerHeight;
      var ratio = span > 0 ? (window.scrollY - start) / span : 0;
      progress.style.transform = 'scaleX(' + Math.min(1, Math.max(0, ratio)) + ')';
    }
    if (!spyTargets.length) return;
    var mark = window.scrollY + window.innerHeight * 0.3;
    var found = spyTargets[0];
    for (var i = 0; i < spyTargets.length; i += 1) {
      if (spyTargets[i].top <= mark) found = spyTargets[i];
      else break;
    }
    if (found !== active) {
      if (active) active.link.classList.remove('is-active');
      found.link.classList.add('is-active');
      active = found;
    }
  }

  var ticking = false;
  function requestScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () { ticking = false; onScroll(); });
  }

  if (outlineLinks.length || progress) {
    measure();
    onScroll();
    window.addEventListener('scroll', requestScroll, { passive: true });
    window.addEventListener('resize', function () { measure(); requestScroll(); });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(measure);
  }

  /* --- Keep the sidebar's current item in view ---------------------- */

  var current = $('.sidebar .nav-section.is-current');
  if (current) {
    var sidebar = $('.sidebar');
    var offset = current.offsetTop - sidebar.clientHeight / 2;
    if (offset > 0) sidebar.scrollTop = offset;
  }

  /* --- Search ------------------------------------------------------ */

  var overlay = $('#search');
  var overlayInput = $('#search-input');
  var overlayResults = $('#search-results');
  var pageInput = $('#search-page-input');
  var pageResults = $('#search-page-results');

  var index = null;
  var indexPromise = null;
  var activeHit = -1;

  function loadIndex() {
    if (indexPromise) return indexPromise;
    indexPromise = fetch(href('assets/search-index.json'))
      .then(function (response) { return response.json(); })
      .then(function (data) {
        index = data.items.map(function (item) {
          var page = data.pages[item[0]];
          var title = item[2] || page[1];
          return {
            title: title,
            url: page[0] + (item[1] ? '#' + item[1] : ''),
            path: page[2] + (page[3] ? ' · ' + page[3] : '') + (item[2] ? ' · ' + page[1] : ''),
            hay: (title + ' ' + page[1] + ' ' + page[2]).toLowerCase(),
          };
        });
        return index;
      })
      .catch(function () { index = []; return index; });
    return indexPromise;
  }

  function tokenise(query) {
    return query.toLowerCase().split(/[^\p{L}\p{N}+#.]+/u).filter(Boolean);
  }

  function search(query, limit) {
    var tokens = tokenise(query);
    if (!tokens.length || !index) return [];
    var out = [];
    for (var i = 0; i < index.length; i += 1) {
      var entry = index[i];
      var score = 0;
      var ok = true;
      for (var t = 0; t < tokens.length; t += 1) {
        var at = entry.hay.indexOf(tokens[t]);
        if (at === -1) { ok = false; break; }
        score += 40 - Math.min(30, at / 4);
        if (at === 0 || /[\s(\/·-]/.test(entry.hay.charAt(at - 1))) score += 22;
        if (entry.title.toLowerCase().indexOf(tokens[t]) !== -1) score += 18;
      }
      if (!ok) continue;
      score -= entry.title.length / 40;
      out.push({ entry: entry, score: score });
    }
    out.sort(function (a, b) { return b.score - a.score; });
    return out.slice(0, limit).map(function (hit) { return hit.entry; });
  }

  function escapeHtml(value) {
    return value.replace(/[&<>"]/g, function (character) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[character];
    });
  }

  function highlight(text, tokens) {
    var safe = escapeHtml(text);
    if (!tokens.length) return safe;
    var pattern = tokens
      .slice()
      .sort(function (a, b) { return b.length - a.length; })
      .map(function (token) { return token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); })
      .join('|');
    try {
      return safe.replace(new RegExp('(' + pattern + ')', 'gi'), '<mark>$1</mark>');
    } catch (e) {
      return safe;
    }
  }

  function render(container, query, limit) {
    var tokens = tokenise(query);
    if (!query.trim()) {
      container.innerHTML = '<p class="search-empty">Start typing to search every question in both guides.</p>';
      return;
    }
    var hits = search(query, limit);
    if (!hits.length) {
      container.innerHTML = '<p class="search-empty">No match for “' + escapeHtml(query) + '”.</p>';
      return;
    }
    container.innerHTML = hits
      .map(function (hit) {
        return (
          '<a class="search-hit" href="' + href(hit.url) + '">' +
          '<span class="search-hit-title">' + highlight(hit.title, tokens) + '</span>' +
          '<span class="search-hit-path">' + escapeHtml(hit.path) + '</span>' +
          '</a>'
        );
      })
      .join('');
    activeHit = -1;
  }

  function openSearch() {
    if (!overlay) return;
    overlay.hidden = false;
    setPrefsOpen(false);
    loadIndex().then(function () { if (overlayInput.value) render(overlayResults, overlayInput.value, 40); });
    render(overlayResults, overlayInput.value, 40);
    overlayInput.focus();
    overlayInput.select();
  }

  function closeSearch() {
    if (overlay) overlay.hidden = true;
  }

  if (overlay) {
    overlay.addEventListener('click', function (event) {
      if (event.target === overlay) closeSearch();
    });
    overlayInput.addEventListener('input', function () {
      loadIndex().then(function () { render(overlayResults, overlayInput.value, 40); });
    });
    overlayInput.addEventListener('keydown', function (event) {
      var hits = $$('.search-hit', overlayResults);
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        if (!hits.length) return;
        if (activeHit >= 0) hits[activeHit].classList.remove('is-active');
        activeHit = event.key === 'ArrowDown'
          ? (activeHit + 1) % hits.length
          : (activeHit - 1 + hits.length) % hits.length;
        hits[activeHit].classList.add('is-active');
        hits[activeHit].scrollIntoView({ block: 'nearest' });
      } else if (event.key === 'Enter') {
        if (activeHit >= 0 && hits[activeHit]) { event.preventDefault(); hits[activeHit].click(); }
        else if (hits[0]) { event.preventDefault(); hits[0].click(); }
      }
    });
  }

  if (pageInput && pageResults) {
    var runPageSearch = function () {
      loadIndex().then(function () { render(pageResults, pageInput.value, 60); });
    };
    pageInput.addEventListener('input', runPageSearch);
    var initial = new URLSearchParams(location.search).get('q');
    if (initial) { pageInput.value = initial; }
    runPageSearch();
    pageInput.focus();
  }

  /* --- Keyboard ----------------------------------------------------- */

  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
      if (overlay && !overlay.hidden) { closeSearch(); return; }
      if (prefsPanel && !prefsPanel.hidden) { setPrefsOpen(false); return; }
      if (body.classList.contains('nav-open')) { setNavOpen(false); return; }
    }

    if ((event.key === 'k' || event.key === 'K') && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      openSearch();
      return;
    }

    if (isTyping(document.activeElement) || event.metaKey || event.ctrlKey || event.altKey) return;

    if (event.key === '/') { event.preventDefault(); openSearch(); return; }

    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      var link = $('[data-nav="' + (event.key === 'ArrowLeft' ? 'prev' : 'next') + '"]');
      if (link) { event.preventDefault(); location.href = link.href; }
    }
  });
})();
