"use client";

import React, { useState, useMemo, useEffect, useLayoutEffect, useRef, useCallback, useSyncExternalStore, startTransition } from 'react';
import { Search, Copy, ChevronDown, X, BookOpen, Menu, Heart, Shuffle, Link2, Eye, Check, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import { motion, AnimatePresence, useReducedMotion, animate } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { toast } from 'sonner';

import rawData from '../data/interview-data.json';

interface Question {
  id: string;
  question: string;
  answer: string;
}

interface Section {
  id: string;
  number: number;
  title: string;
  questions: Question[];
}

interface Data {
  sections: Section[];
  stats: {
    totalSections: number;
    totalQuestions: number;
  };
}

const data = rawData as Data;
const AUTO_EXPAND_SEARCH_LIMIT = 12;

const MODAL_SPRING = { type: 'spring', stiffness: 300, damping: 34, mass: 0.8 } as const;
const RAIL_SPRING = { type: 'spring', stiffness: 380, damping: 38 } as const;
const EASE_OUT = [0.22, 1, 0.36, 1] as const;

function loadStoredSet(key: string): Set<string> {
  if (typeof window === 'undefined') return new Set();
  try {
    const saved = localStorage.getItem(key);
    if (saved) return new Set(JSON.parse(saved));
  } catch {
    // ignore invalid stored data
  }
  return new Set();
}

function loadStoredBoolean(key: string, fallback = false): boolean {
  if (typeof window === 'undefined') return fallback;
  try {
    const saved = localStorage.getItem(key);
    if (saved !== null) return saved === 'true';
  } catch {
    // ignore invalid stored data
  }
  return fallback;
}

function safeSetItem(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // storage may be full or unavailable — persistence is best-effort
  }
}

/* Mono numeral that counts up on first mount, then tracks value changes.
   With animateIn=false (intro already played) it renders settled. */
function CountUp({ value, suffix = '', animateIn = true }: { value: number; suffix?: string; animateIn?: boolean }) {
  const reduce = useReducedMotion();
  const ref = useRef<HTMLSpanElement>(null);
  const prev = useRef<number | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (reduce || (prev.current === null && !animateIn)) {
      el.textContent = `${value}${suffix}`;
      prev.current = value;
      return;
    }
    const from = prev.current ?? 0;
    const firstRun = prev.current === null;
    prev.current = value;
    if (from === value) {
      el.textContent = `${value}${suffix}`;
      return;
    }
    const controls = animate(from, value, {
      duration: firstRun ? 0.9 : 0.4,
      ease: EASE_OUT,
      onUpdate: (v) => {
        el.textContent = `${Math.round(v)}${suffix}`;
      },
    });
    return () => controls.stop();
  }, [value, suffix, reduce, animateIn]);

  return (
    <span ref={ref} className={animateIn ? 'blur-settle' : undefined}>
      {value}
      {suffix}
    </span>
  );
}

/* "Paper first, ink second": surface opens via grid-rows, content fades up.
   Mounts on expand (double rAF so the collapsed frame commits before the
   transition starts), unmounts after the collapse transition ends. */
function AnswerReveal({ open, children }: { open: boolean; children: React.ReactNode }) {
  const [mounted, setMounted] = useState(open);
  const ref = useRef<HTMLDivElement>(null);
  const isShown = useRef(false);

  // Derive mount state from props during render (official React pattern)
  if (open && !mounted) {
    setMounted(true);
  }

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;

    if (open) {
      if (isShown.current) return;
      isShown.current = true;
      let raf2 = 0;
      const raf1 = requestAnimationFrame(() => {
        raf2 = requestAnimationFrame(() => el.classList.add('is-open'));
      });
      return () => {
        cancelAnimationFrame(raf1);
        if (raf2) cancelAnimationFrame(raf2);
      };
    }

    if (!isShown.current) return;
    isShown.current = false;
    el.classList.remove('is-open');
    const timer = window.setTimeout(() => setMounted(false), 350);
    const onEnd = (e: TransitionEvent) => {
      if (e.target === el && e.propertyName === 'grid-template-rows') {
        window.clearTimeout(timer);
        setMounted(false);
      }
    };
    el.addEventListener('transitionend', onEnd);
    return () => {
      el.removeEventListener('transitionend', onEnd);
      window.clearTimeout(timer);
    };
  }, [open, mounted]);

  if (!mounted) return null;

  return (
    <div ref={ref} className="answer-reveal">
      <div className="answer-reveal-inner">
        <div className="answer-fade">{children}</div>
      </div>
    </div>
  );
}

export default function AIInterviewPrep() {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchFocused, setSearchFocused] = useState(false);
  const [activeSection, setActiveSection] = useState<string | null>(null);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [showMobileToc, setShowMobileToc] = useState(false);
  const [favorites, setFavorites] = useState<Set<string>>(() =>
    loadStoredSet('ai-interview-favorites')
  );
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
  const [focusQuestion, setFocusQuestion] = useState<Question | null>(null);
  const [viewedQuestions, setViewedQuestions] = useState<Set<string>>(() =>
    loadStoredSet('ai-interview-viewed')
  );
  const isMounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  );
  const reduceMotion = useReducedMotion();

  const headerRef = useRef<HTMLElement>(null);
  const lastScrollY = useRef(0);
  const scrollOffsetRef = useRef(124);
  const [filtersCollapsed, setFiltersCollapsed] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() =>
    loadStoredBoolean('ai-interview-sidebar-collapsed')
  );
  // User-collapsed cards while search auto-expand is active
  const [searchCollapsedIds, setSearchCollapsedIds] = useState<Set<string>>(new Set());
  // Once the opening choreography has played, remounts render settled.
  // Flips via timer/interaction only, so SSR and hydration renders match.
  const [introDone, setIntroDone] = useState(false);

  useEffect(() => {
    const t = window.setTimeout(() => setIntroDone(true), 1600);
    return () => window.clearTimeout(t);
  }, []);

  // Track sticky header height for sidebar offset and scroll targets.
  // Only a CSS var + ref update — no React state, so the 300ms header
  // collapse animation never re-renders the 456-card tree.
  useEffect(() => {
    const header = headerRef.current;
    if (!header) return;

    const update = () => {
      const height = Math.ceil(header.getBoundingClientRect().height);
      scrollOffsetRef.current = height + 12;
      document.documentElement.style.setProperty('--site-header-height', `${height}px`);
    };

    update();
    const observer = new ResizeObserver(update);
    observer.observe(header);
    window.addEventListener('resize', update);

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', update);
    };
  }, []);

  // Multi-pass scroll: content-visibility placeholders settle to real
  // heights as the destination renders, so re-measure across two frames.
  const scrollToElementId = useCallback((elId: string) => {
    const go = () => {
      const el = document.getElementById(elId);
      if (!el) return;
      const top = el.getBoundingClientRect().top + window.scrollY - scrollOffsetRef.current;
      window.scrollTo({ top, behavior: 'auto' });
    };
    go();
    requestAnimationFrame(() => {
      go();
      requestAnimationFrame(go);
    });
  }, []);

  // Collapse filter rows on scroll down; expand on scroll up or near top
  useEffect(() => {
    lastScrollY.current = window.scrollY;

    const handleScroll = () => {
      const currentY = window.scrollY;
      const delta = currentY - lastScrollY.current;

      if (currentY <= 48) {
        setFiltersCollapsed(false);
      } else if (delta > 6) {
        setFiltersCollapsed(true);
      } else if (delta < -6) {
        setFiltersCollapsed(false);
      }

      lastScrollY.current = currentY;
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Lock page scroll when modal or mobile nav is open
  useEffect(() => {
    const shouldLock = Boolean(focusQuestion) || showMobileToc;
    if (!shouldLock) return;

    const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth;
    const previousOverflow = document.body.style.overflow;
    const previousPaddingRight = document.body.style.paddingRight;

    document.body.style.overflow = 'hidden';
    if (scrollbarWidth > 0) {
      document.body.style.paddingRight = `${scrollbarWidth}px`;
    }

    return () => {
      document.body.style.overflow = previousOverflow;
      document.body.style.paddingRight = previousPaddingRight;
    };
  }, [focusQuestion, showMobileToc]);

  // Persist favorites
  useEffect(() => {
    safeSetItem('ai-interview-favorites', JSON.stringify(Array.from(favorites)));
  }, [favorites]);

  useEffect(() => {
    safeSetItem('ai-interview-viewed', JSON.stringify(Array.from(viewedQuestions)));
  }, [viewedQuestions]);

  useEffect(() => {
    safeSetItem('ai-interview-sidebar-collapsed', String(sidebarCollapsed));
  }, [sidebarCollapsed]);

  // Deep link support: open specific question from URL hash.
  // Runs on mount and real hashchange only — never on re-render.
  useEffect(() => {
    let timer = 0;
    const handleHash = () => {
      const hash = window.location.hash.replace('#', '');
      if (!hash) return;
      for (const section of data.sections) {
        const found = section.questions.find(q => q.id === hash);
        if (found) {
          setExpandedIds(prev => new Set(prev).add(hash));
          window.clearTimeout(timer);
          timer = window.setTimeout(() => scrollToElementId(`card-${hash}`), 80);
          break;
        }
      }
    };

    handleHash();
    window.addEventListener('hashchange', handleHash);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener('hashchange', handleHash);
    };
  }, [scrollToElementId]);

  // Mark question as viewed when expanded
  const markAsViewed = (id: string) => {
    if (!viewedQuestions.has(id)) {
      setViewedQuestions(prev => {
        const next = new Set(prev);
        next.add(id);
        return next;
      });
    }
  };

  const closeFocus = () => {
    setFocusQuestion(null);
  };

  const copyQA = (q: Question) => {
    const text = `${q.question}\n\n${q.answer}`;
    navigator.clipboard.writeText(text).then(() => {
      toast.success('Copied to clipboard');
    });
  };

  const copyQuestion = (q: Question) => {
    navigator.clipboard.writeText(q.question).then(() => {
      toast.success('Question copied');
    });
  };

  // Keyboard support for modal — plain keys only, never the native
  // copy/paste shortcuts (Cmd/Ctrl+C must keep the user's selection)
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 'Escape' && focusQuestion) {
        closeFocus();
      }
      if (focusQuestion && e.key.toLowerCase() === 'c') {
        if (document.activeElement?.tagName !== 'INPUT') {
          copyQA(focusQuestion);
        }
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [focusQuestion]);

  // Simple search highlight
  const highlightText = (text: string, term: string): React.ReactNode => {
    if (!term.trim()) return text;

    const regex = new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    const parts = text.split(regex);

    return parts.map((part, i) =>
      regex.test(part) ? (
        <mark key={i}>{part}</mark>
      ) : (
        <React.Fragment key={i}>{part}</React.Fragment>
      )
    );
  };

  // Generate a short preview of the answer for collapsed state.
  // While searching, center the preview on the first match so every
  // result card shows WHY it matched.
  const getAnswerPreview = (answer: string, term?: string): string => {
    const clean = (s: string) => s.replace(/[#*`]/g, '').replace(/\s+/g, ' ').trim();
    const t = term?.trim().toLowerCase();
    if (t) {
      const cleaned = clean(answer);
      const i = cleaned.toLowerCase().indexOf(t);
      if (i >= 0) {
        const start = Math.max(0, i - 70);
        let snippet = cleaned.slice(start, start + 180).trim();
        if (start > 0) snippet = '…' + snippet;
        if (start + 180 < cleaned.length) snippet += '…';
        return snippet;
      }
    }
    let preview = clean(answer.split('\n\n')[0]);
    if (preview.length > 180) {
      preview = preview.slice(0, 177) + '…';
    }
    return preview;
  };

  // Global search + favorites filter
  const filteredSections = useMemo(() => {
    let sections = data.sections;

    const term = searchTerm.toLowerCase().trim();
    if (term) {
      sections = sections
        .map((section) => {
          const matchingQs = section.questions.filter(
            (q) =>
              q.question.toLowerCase().includes(term) ||
              q.answer.toLowerCase().includes(term)
          );
          return { ...section, questions: matchingQs };
        })
        .filter((s) => s.questions.length > 0);
    }

    if (showFavoritesOnly) {
      sections = sections
        .map((section) => ({
          ...section,
          questions: section.questions.filter((q) => favorites.has(q.id)),
        }))
        .filter((s) => s.questions.length > 0);
    }

    return sections;
  }, [searchTerm, showFavoritesOnly, favorites]);

  // Auto-expand all matching results when searching
  const searchExpandedIds = useMemo(() => {
    if (!searchTerm.trim()) return null;
    const matchCount = filteredSections.reduce((sum, s) => sum + s.questions.length, 0);
    if (matchCount > AUTO_EXPAND_SEARCH_LIMIT) return null;

    const allMatching = new Set<string>();
    filteredSections.forEach((s) =>
      s.questions.forEach((q) => allMatching.add(q.id))
    );
    return allMatching;
  }, [searchTerm, filteredSections]);

  // Auto-expanded search results minus the ones the user collapsed
  const effectiveExpandedIds = useMemo(() => {
    if (!searchExpandedIds) return expandedIds;
    if (searchCollapsedIds.size === 0) return searchExpandedIds;
    const s = new Set(searchExpandedIds);
    searchCollapsedIds.forEach((id) => s.delete(id));
    return s;
  }, [searchExpandedIds, expandedIds, searchCollapsedIds]);

  // Single entry point for search changes: resets per-search collapse
  // state and marks the opening choreography as done.
  const updateSearchTerm = (value: string) => {
    setIntroDone(true);
    setSearchCollapsedIds((prev) => (prev.size === 0 ? prev : new Set()));
    setSearchTerm(value);
  };

  const visibleQuestionCount = useMemo(() => {
    return filteredSections.reduce((sum, s) => sum + s.questions.length, 0);
  }, [filteredSections]);

  const viewedCount = useMemo(() => viewedQuestions.size, [viewedQuestions]);
  const progressPercentage = Math.min(
    100,
    Math.round((viewedCount / data.stats.totalQuestions) * 100)
  );
  const displayedViewedCount = isMounted ? viewedCount : 0;
  const displayedProgressPercentage = isMounted ? progressPercentage : 0;

  const expandedCount = useMemo(() => {
    return Array.from(effectiveExpandedIds).filter((id) =>
      filteredSections.some((s) => s.questions.some((q) => q.id === id))
    ).length;
  }, [effectiveExpandedIds, filteredSections]);

  const toggleQuestion = (id: string) => {
    // While search auto-expand is active, toggling works against the
    // per-search collapsed set so cards actually close.
    if (searchExpandedIds?.has(id)) {
      setSearchCollapsedIds((prev) => {
        const next = new Set(prev);
        if (next.has(id)) {
          next.delete(id);
          markAsViewed(id);
        } else {
          next.add(id);
        }
        return next;
      });
      return;
    }
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
        markAsViewed(id);
      }
      return next;
    });
  };

  const openFocus = (q: Question, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    focusReturnRef.current = document.activeElement as HTMLElement | null;
    markAsViewed(q.id);
    setFocusQuestion(q);
    setExpandedIds(prev => new Set(prev).add(q.id));
  };

  const shareQuestion = (q: Question, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    const url = `${window.location.origin}${window.location.pathname}#${q.id}`;
    navigator.clipboard.writeText(url).then(() => {
      window.history.pushState(null, '', `#${q.id}`);
      toast.success('Link copied to clipboard');
    });
  };

  const toggleFavorite = (id: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setFavorites((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
        toast.success('Added to favorites');
      }
      return next;
    });
  };

  // startTransition keeps the click responsive while 456 answers mount
  const expandAll = () => {
    if (searchExpandedIds) {
      setSearchCollapsedIds(new Set());
    }
    const allIds = new Set<string>();
    filteredSections.forEach((s) =>
      s.questions.forEach((q) => allIds.add(q.id))
    );
    startTransition(() => setExpandedIds(allIds));
  };

  const collapseAll = () => {
    if (searchExpandedIds) {
      setSearchCollapsedIds(new Set(searchExpandedIds));
      return;
    }
    startTransition(() => setExpandedIds(new Set()));
  };

  const goToRandomQuestion = () => {
    const allQuestions = filteredSections.flatMap(s => s.questions);
    if (allQuestions.length === 0) return;

    const randomQ = allQuestions[Math.floor(Math.random() * allQuestions.length)];

    setExpandedIds(prev => new Set(prev).add(randomQ.id));
    setSearchCollapsedIds((prev) => {
      if (!prev.has(randomQ.id)) return prev;
      const next = new Set(prev);
      next.delete(randomQ.id);
      return next;
    });
    markAsViewed(randomQ.id);

    setTimeout(() => scrollToElementId(`card-${randomQ.id}`), 50);

    toast.success('Random question loaded');
  };

  const toggleSection = (sectionId: string, expand: boolean) => {
    const section = data.sections.find(s => s.id === sectionId);
    if (!section) return;

    startTransition(() =>
      setExpandedIds(prev => {
        const next = new Set(prev);
        section.questions.forEach(q => {
          if (expand) {
            next.add(q.id);
          } else {
            next.delete(q.id);
          }
        });
        return next;
      })
    );

    if (expand) {
      setViewedQuestions(prev => {
        const next = new Set(prev);
        section.questions.forEach(q => next.add(q.id));
        return next;
      });
    }
  };

  const scrollToSection = (sectionId: string) => {
    scrollToElementId(sectionId);
    setActiveSection(sectionId);
    setShowMobileToc(false);
  };

  // Keyboard shortcut: press / to focus search. Escape clears the
  // search only when no overlay is open (the overlay consumes it).
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === '/' && document.activeElement?.tagName !== 'INPUT') {
        e.preventDefault();
        const input = document.getElementById('search-input') as HTMLInputElement;
        input?.focus();
      }
      if (e.key === 'Escape' && searchTerm && !focusQuestion && !showMobileToc) {
        setSearchCollapsedIds(new Set());
        setSearchTerm('');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [searchTerm, focusQuestion, showMobileToc]);

  // Track active section on scroll. Sections are far taller than the
  // viewport, so intersection ratios never reach usable thresholds —
  // instead, pick the last section whose top has passed the header on
  // each (rAF-throttled) scroll. 14 rect reads per frame is cheap.
  useEffect(() => {
    let ticking = false;
    const compute = () => {
      ticking = false;
      const offset = scrollOffsetRef.current + 24;
      let current: string | null = null;
      for (const s of data.sections) {
        const el = document.getElementById(s.id);
        if (!el) continue;
        if (el.getBoundingClientRect().top <= offset) {
          current = s.id;
        } else {
          break;
        }
      }
      if (current) {
        setActiveSection((prev) => (prev === current ? prev : current));
      }
    };
    const onScroll = () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(compute);
      }
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    const raf = requestAnimationFrame(compute);
    return () => {
      window.removeEventListener('scroll', onScroll);
      cancelAnimationFrame(raf);
    };
  }, []);

  // Move focus into the dialog on open; restore it on close
  const modalCloseRef = useRef<HTMLButtonElement>(null);
  const drawerCloseRef = useRef<HTMLButtonElement>(null);
  const focusReturnRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (focusQuestion) {
      modalCloseRef.current?.focus();
      return;
    }
    focusReturnRef.current?.focus();
    focusReturnRef.current = null;
  }, [focusQuestion]);

  useEffect(() => {
    if (!showMobileToc) return;
    drawerCloseRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setShowMobileToc(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [showMobileToc]);

  const isSearching = searchTerm.trim().length > 0;
  const showEntrance = !isSearching && !showFavoritesOnly && !reduceMotion && !introDone;
  // Gate localStorage-derived layout behind mount to keep SSR HTML valid
  const shownSidebarCollapsed = isMounted && sidebarCollapsed;

  const focusSection = focusQuestion
    ? data.sections.find((s) => s.questions.some((q) => q.id === focusQuestion.id))
    : null;
  const focusReadMinutes = focusQuestion
    ? Math.max(1, Math.ceil(focusQuestion.answer.split(/\s+/).length / 220))
    : 0;

  const heroStagger = {
    hidden: {},
    show: { transition: { staggerChildren: 0.06 } },
  };
  const heroChild = {
    hidden: reduceMotion
      ? { opacity: 0 }
      : { opacity: 0, y: 14, filter: 'blur(5px)' },
    show: reduceMotion
      ? { opacity: 1, transition: { duration: 0.2 } }
      : { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.7, ease: EASE_OUT } },
  };

  return (
    <div className="min-h-screen bg-page text-ink-mid">
      {/* Page chrome is inert while the focus modal is open */}
      <div inert={focusQuestion ? true : undefined}>
      {/* Top nav / header */}
      <header ref={headerRef} className="site-header sticky top-0 z-50">
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6">
          <div className="flex h-16 items-center justify-between gap-4">
            <div className="flex items-center gap-3.5">
              <div className="brand-tile">
                <div className="brand-glow" aria-hidden="true" />
                <BookOpen className="relative h-[18px] w-[18px]" />
              </div>
              <div>
                <div className="brand-name">AI Engineering</div>
                <div className="kicker -mt-0.5">Interview Prep</div>
              </div>
            </div>

            <div className="flex items-center gap-2 text-sm">
              <a href="/concept-map" className="btn-soft hidden !h-9 text-xs sm:inline-flex">
                <Link2 className="h-3.5 w-3.5" />
                <span>Concept map</span>
              </a>
              <div className="stat-chip hidden sm:flex">
                <span className="num">{data.stats.totalQuestions}</span>
                <span>questions</span>
                <span className="opacity-40">·</span>
                <span className="num">{data.stats.totalSections}</span>
                <span>sections</span>
              </div>
              <div className="stat-chip progress-chip hidden md:flex">
                <Check className="h-3.5 w-3.5" />
                <span className="num">{displayedProgressPercentage}%</span>
                <span>viewed</span>
              </div>
            </div>
          </div>
        </div>

        {/* Search bar */}
        <div className="border-t border-line-soft">
          <div className={`mx-auto max-w-7xl px-4 sm:px-6 transition-[padding] duration-300 ${filtersCollapsed ? 'py-2.5' : 'py-4'}`}>
            <div className="flex flex-col items-stretch gap-3 sm:flex-row sm:items-center">
              <div className="relative flex-1">
                <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-low" />
                <input
                  id="search-input"
                  type="text"
                  value={searchTerm}
                  onChange={(e) => updateSearchTerm(e.target.value)}
                  onFocus={() => setSearchFocused(true)}
                  onBlur={() => setSearchFocused(false)}
                  placeholder="Search questions and answers..."
                  aria-label="Search questions and answers"
                  className="search-input w-full pl-11 pr-11"
                />
                <div className="search-slot">
                  {searchTerm ? (
                    <button
                      onClick={() => {
                        updateSearchTerm('');
                        document.getElementById('search-input')?.focus();
                      }}
                      className="search-clear"
                      aria-label="Clear search"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  ) : !searchFocused ? (
                    <kbd aria-hidden="true">/</kbd>
                  ) : null}
                </div>
              </div>

              <div className="flex flex-wrap items-center justify-end gap-2">
                <button
                  onClick={goToRandomQuestion}
                  className="btn-primary"
                  title="Load a random question (great for practice)"
                >
                  <Shuffle className="h-4 w-4" />
                  <span className="hidden sm:inline">Random</span>
                </button>
                <button onClick={expandAll} className="btn-soft">
                  <ChevronDown className="h-4 w-4" /> <span className="hidden sm:inline">Expand</span>
                </button>
                <button onClick={collapseAll} className="btn-soft">
                  <ChevronDown className="h-4 w-4 rotate-180" /> <span className="hidden sm:inline">Collapse</span>
                </button>
                <button
                  onClick={() => setShowMobileToc(!showMobileToc)}
                  className="btn-soft !px-3 sm:hidden"
                  aria-label="Open sections"
                  aria-expanded={showMobileToc}
                >
                  <Menu className="h-4 w-4" />
                </button>
              </div>
            </div>

            {/* Compact active-filter hint when collapsed */}
            {filtersCollapsed && (searchTerm || showFavoritesOnly) && (
              <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                {showFavoritesOnly && <span className="filter-pill">Favorites only</span>}
                {searchTerm && (
                  <span className="filter-pill">
                    Search: “{searchTerm.length > 24 ? `${searchTerm.slice(0, 24)}…` : searchTerm}”
                  </span>
                )}
                <button
                  onClick={() => {
                    updateSearchTerm('');
                    setShowFavoritesOnly(false);
                  }}
                  className="text-btn underline underline-offset-4"
                >
                  Clear
                </button>
              </div>
            )}

            {/* Results summary + Filters — collapses on scroll */}
            <div className={`header-filters ${filtersCollapsed ? 'header-filters-collapsed' : ''}`}>
              <div className="header-filters-inner">
                <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
                  <div className="flex items-center gap-2.5">
                    {isSearching ? (
                      <span>
                        <span className="font-medium text-ink-hi">{visibleQuestionCount}</span>
                        <span className="text-ink-low"> result{visibleQuestionCount === 1 ? '' : 's'} for “{searchTerm}”</span>
                      </span>
                    ) : showFavoritesOnly ? (
                      <span className="font-medium text-ink-hi">
                        {visibleQuestionCount} favorite{visibleQuestionCount === 1 ? '' : 's'}
                      </span>
                    ) : (
                      <span className="text-ink-low">
                        Showing <span className="text-ink-mid">{visibleQuestionCount}</span> / {data.stats.totalQuestions}
                      </span>
                    )}

                    {expandedCount > 0 && <span className="stat-pill">{expandedCount} open</span>}
                  </div>

                  {/* Favorites toggle */}
                  <button
                    onClick={() => {
                      setIntroDone(true);
                      setShowFavoritesOnly(!showFavoritesOnly);
                    }}
                    aria-pressed={showFavoritesOnly}
                    className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-colors duration-200 ${
                      showFavoritesOnly
                        ? 'border-line-strong bg-surface-3 text-ink-hi'
                        : 'border-line-soft bg-surface-2 text-ink-low hover:border-line-strong hover:text-ink-mid'
                    }`}
                  >
                    <Heart className={`h-3.5 w-3.5 ${showFavoritesOnly ? 'fill-current text-ember' : ''}`} />
                    Favorites
                    {isMounted && favorites.size > 0 && (
                      <span className="ml-0.5 rounded bg-page/60 px-1.5 font-mono tabular-nums">{favorites.size}</span>
                    )}
                  </button>

                  {(searchTerm || showFavoritesOnly) && (
                    <button
                      onClick={() => {
                        updateSearchTerm('');
                        setShowFavoritesOnly(false);
                      }}
                      className="text-btn underline underline-offset-4"
                    >
                      Clear all filters
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Ambient session progress — pinned to the header's bottom edge */}
        <div
          className="header-progress"
          style={{ transform: `scaleX(${displayedProgressPercentage / 100})` }}
          aria-hidden="true"
        />
      </header>

      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <div className="flex flex-col gap-8 pt-6 pb-24 lg:flex-row">
          {/* Sidebar TOC */}
          <aside
            className={`sidebar-aside hidden shrink-0 lg:block ${shownSidebarCollapsed ? 'w-12' : 'w-72'}`}
          >
            <div
              className="toc-panel sticky overflow-y-auto overscroll-contain p-2"
              style={{
                top: 'calc(var(--site-header-height, 7rem) + 0.75rem)',
                maxHeight: 'calc(100vh - var(--site-header-height, 7rem) - 1.75rem)',
              }}
            >
              <div
                className={`mb-3 flex items-center ${
                  shownSidebarCollapsed ? 'justify-center px-1' : 'justify-between px-2'
                }`}
              >
                {!shownSidebarCollapsed && <div className="kicker px-1 pt-1">Contents</div>}
                <button
                  type="button"
                  onClick={() => setSidebarCollapsed((prev) => !prev)}
                  className="icon-btn"
                  title={shownSidebarCollapsed ? 'Expand contents sidebar' : 'Collapse contents sidebar'}
                  aria-label={shownSidebarCollapsed ? 'Expand contents sidebar' : 'Collapse contents sidebar'}
                  aria-expanded={!shownSidebarCollapsed}
                >
                  {shownSidebarCollapsed ? (
                    <PanelLeftOpen className="h-4 w-4" />
                  ) : (
                    <PanelLeftClose className="h-4 w-4" />
                  )}
                </button>
              </div>

              {!shownSidebarCollapsed ? (
                <>
                  <nav className="toc-nav-full space-y-0.5">
                    {data.sections.map((section) => {
                      const originalCount = section.questions.length;
                      const isActive = activeSection === section.id;

                      let displayCount = originalCount;
                      if (showFavoritesOnly) {
                        displayCount = section.questions.filter((q) => favorites.has(q.id)).length;
                      }

                      const viewedInSection = section.questions.filter((q) => viewedQuestions.has(q.id)).length;
                      const sectionProgress = isMounted
                        ? Math.round((viewedInSection / originalCount) * 100)
                        : 0;

                      return (
                        <button
                          key={section.id}
                          onClick={() => scrollToSection(section.id)}
                          className={`toc-link flex w-full items-center justify-between px-3 py-2 text-left text-[13.5px] ${
                            isActive ? 'active' : ''
                          }`}
                        >
                          {isActive && (
                            <motion.div
                              layoutId="toc-rail"
                              className="toc-rail"
                              transition={reduceMotion ? { duration: 0 } : RAIL_SPRING}
                            />
                          )}
                          <span className="truncate pr-3">
                            {section.number}. {section.title}
                          </span>
                          {showFavoritesOnly ? (
                            <span className="shrink-0 font-mono text-[10px] tabular-nums text-ink-low">
                              {displayCount}
                            </span>
                          ) : sectionProgress >= 100 ? (
                            <Check className="h-3 w-3 shrink-0 text-sage-400" aria-label="Section complete" />
                          ) : (
                            <span className="toc-track">
                              <span
                                className="toc-track-fill block"
                                style={{ transform: `scaleX(${sectionProgress / 100})` }}
                              />
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </nav>

                  <div className="mt-8 px-3 pb-2 text-[11px] leading-relaxed text-ink-low">
                    Press <kbd>/</kbd> to search
                    <br />
                    <span className="opacity-70">Click cards to expand · Heart to favorite</span>
                  </div>
                </>
              ) : (
                <nav className="toc-nav-mini flex flex-col items-center gap-1 px-1 pb-1">
                  {data.sections.map((section) => {
                    const isActive = activeSection === section.id;

                    return (
                      <button
                        key={section.id}
                        onClick={() => scrollToSection(section.id)}
                        title={`${section.number}. ${section.title}`}
                        className={`toc-mini ${isActive ? 'active' : ''}`}
                      >
                        {section.number}
                      </button>
                    );
                  })}
                </nav>
              )}
            </div>
          </aside>

          {/* Mobile TOC drawer */}
          <AnimatePresence>
            {showMobileToc && (
              <motion.div
                key="drawer-overlay"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="focus-overlay fixed inset-0 z-[60] lg:hidden"
                onClick={() => setShowMobileToc(false)}
              >
                <motion.div
                  key="drawer-panel"
                  initial={reduceMotion ? { opacity: 0 } : { x: '-100%' }}
                  animate={reduceMotion ? { opacity: 1 } : { x: 0 }}
                  exit={reduceMotion ? { opacity: 0 } : { x: '-100%' }}
                  transition={reduceMotion ? { duration: 0.2 } : MODAL_SPRING}
                  role="dialog"
                  aria-modal="true"
                  aria-label="Sections"
                  className="drawer-panel absolute bottom-0 left-0 top-0 w-[min(18rem,85vw)] max-w-full overflow-y-auto overscroll-contain p-4"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="mb-4 flex items-center justify-between">
                    <div className="kicker">Sections</div>
                    <button
                      ref={drawerCloseRef}
                      onClick={() => setShowMobileToc(false)}
                      className="icon-btn"
                      aria-label="Close sections"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                  <div className="space-y-1">
                    {data.sections.map((section) => (
                      <button
                        key={section.id}
                        onClick={() => scrollToSection(section.id)}
                        className="toc-link flex w-full items-center justify-between px-3 py-2.5 text-left text-sm"
                      >
                        <span>
                          {section.number}. {section.title}
                        </span>
                        <span className="font-mono text-xs tabular-nums text-ink-low">{section.questions.length}</span>
                      </button>
                    ))}
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Main content */}
          <main className="min-w-0 flex-1">
            {/* Keep an h1 in the tree while the hero is filtered away */}
            {(isSearching || showFavoritesOnly) && (
              <h1 className="sr-only">AI Engineering Interview Prep</h1>
            )}
            {/* Intro banner */}
            {!isSearching && !showFavoritesOnly && (
              <motion.div
                variants={heroStagger}
                initial={introDone ? false : 'hidden'}
                animate="show"
                className={`hero-card mb-10 px-6 py-8 sm:px-10 sm:py-10 ${introDone ? 'no-intro' : ''}`}
              >
                <div className="grid gap-8 xl:grid-cols-[minmax(0,1fr)_20rem] xl:items-end">
                  <div>
                    <motion.h1 variants={heroChild} className="hero-title max-w-2xl text-balance">
                      Practice high-signal AI engineering interview questions.
                    </motion.h1>
                    <motion.div variants={heroChild}>
                      <div className="hero-rule mt-5" aria-hidden="true" />
                      <p className="mt-5 max-w-2xl text-[15px] leading-7 text-ink-mid">
                        Search, drill random prompts, mark favorites, and use focus mode across LLMs, RAG,
                        Agents, Fine-Tuning, Vector DBs, System Design, LLMOps, Evaluation, Safety,
                        Infrastructure, and more.
                      </p>
                    </motion.div>
                    <motion.div variants={heroChild} className="mt-6 flex flex-wrap gap-2 text-xs">
                      <div className="hero-chip">{data.stats.totalQuestions} curated questions</div>
                      <div className="hero-chip">Favorites saved locally</div>
                      <div className="hero-chip">Random drill mode</div>
                    </motion.div>
                  </div>
                  <motion.div variants={heroChild} className="grid grid-cols-3 gap-2 sm:gap-3 xl:grid-cols-1">
                    <div className="stat-tile">
                      <span className="label">Viewed</span>
                      <span className="value">
                        <CountUp value={displayedViewedCount} animateIn={!introDone} />
                      </span>
                    </div>
                    <div className="stat-tile">
                      <span className="label">Favorites</span>
                      <span className="value">
                        <CountUp value={isMounted ? favorites.size : 0} animateIn={!introDone} />
                      </span>
                    </div>
                    <div className="stat-tile">
                      <span className="label">Progress</span>
                      <span className="value">
                        <CountUp value={displayedProgressPercentage} suffix="%" animateIn={!introDone} />
                      </span>
                      <div className="stat-track">
                        <div
                          className="stat-fill"
                          style={{ transform: `scaleX(${displayedProgressPercentage / 100})` }}
                        />
                      </div>
                    </div>
                  </motion.div>
                </div>
              </motion.div>
            )}

            {filteredSections.length === 0 && (
              <div className="hero-card p-10 text-center">
                {showFavoritesOnly ? (
                  <>
                    <Heart className="mx-auto mb-3 h-8 w-8 text-ink-low opacity-60" />
                    <p className="text-lg text-ink-hi">No favorites yet</p>
                    <p className="mt-1 text-sm text-ink-low">
                      Click the heart icon on any question to save it for later.
                    </p>
                    <button
                      onClick={() => setShowFavoritesOnly(false)}
                      className="text-btn mt-4 !text-sm underline underline-offset-4"
                    >
                      Browse all questions →
                    </button>
                  </>
                ) : (
                  <>
                    <p className="text-lg text-ink-hi">No matches found.</p>
                    <p className="mt-1 text-sm text-ink-low">Try different keywords or clear your filters.</p>
                  </>
                )}
              </div>
            )}

            {filteredSections.map((section, sectionIdx) => (
              <div key={section.id} id={section.id} className="section-block mb-24">
                {/* Section header */}
                <div className="section-plate mb-5">
                  <div className="ghost-folio" aria-hidden="true">
                    {String(section.number).padStart(2, '0')}
                  </div>
                  <div className="flex flex-col gap-3 pb-4 sm:flex-row sm:items-end sm:justify-between">
                    <div className="min-w-0">
                      <div className="kicker section-kicker mb-1.5">
                        Section {String(section.number).padStart(2, '0')} · {section.questions.length} questions
                      </div>
                      <h2 className="section-title break-words">{section.title}</h2>
                    </div>

                    <div className="flex shrink-0 items-center gap-4 self-start sm:self-auto">
                      <button onClick={() => toggleSection(section.id, true)} className="text-btn">
                        Expand all
                      </button>
                      <button onClick={() => toggleSection(section.id, false)} className="text-btn">
                        Collapse
                      </button>
                    </div>
                  </div>
                  <div className="section-rule" />
                </div>

                {/* Questions */}
                <div className="space-y-4">
                  {section.questions.map((q, idx) => {
                    const isExpanded = effectiveExpandedIds.has(q.id);
                    const isFavorited = favorites.has(q.id);
                    const enters = showEntrance && sectionIdx === 0 && idx < 6;

                    return (
                      <div
                        id={`card-${q.id}`}
                        key={q.id}
                        className={`question-card group overflow-hidden ${isExpanded ? 'expanded' : ''} ${
                          enters ? 'card-enter' : ''
                        }`}
                        style={enters ? { animationDelay: `${350 + idx * 40}ms` } : undefined}
                      >
                        <div className="card-spine" aria-hidden="true" />
                        {/* Question header — the title is a real <button>
                            (accordion pattern) so keyboard users can operate
                            the row AND the action buttons independently */}
                        <div
                          onClick={() => toggleQuestion(q.id)}
                          className="question-header w-full cursor-pointer px-4 py-4 text-left sm:px-6 sm:py-5"
                        >
                          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:gap-4">
                            <div className="flex min-w-0 items-start gap-3 sm:contents">
                              <div className="mt-0.5 shrink-0">
                                <div className="q-index">
                                  {idx + 1}
                                  {isMounted && viewedQuestions.has(q.id) && (
                                    <>
                                      <span className="viewed-dot" aria-hidden="true" />
                                      <span className="sr-only">viewed</span>
                                    </>
                                  )}
                                </div>
                              </div>

                              <div className="min-w-0 flex-1 pr-1">
                                <h3 className="question-text break-words">
                                  <button
                                    type="button"
                                    className="question-toggle"
                                    aria-expanded={isExpanded}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      toggleQuestion(q.id);
                                    }}
                                  >
                                    {highlightText(q.question, searchTerm)}
                                  </button>
                                </h3>

                                {!isExpanded && (
                                  <div className="question-preview">
                                    {highlightText(getAnswerPreview(q.answer, searchTerm), searchTerm)}
                                  </div>
                                )}
                              </div>
                            </div>

                            <div className="flex shrink-0 items-center justify-end gap-0.5 border-t border-line-soft pt-3 sm:border-0 sm:pt-0.5">
                              <button
                                onClick={(e) => openFocus(q, e)}
                                className="icon-btn"
                                title="Open in focus mode for easier reading"
                              >
                                <Eye className="h-4 w-4" />
                              </button>

                              <button
                                onClick={(e) => shareQuestion(q, e)}
                                className="icon-btn"
                                title="Copy direct link to this question"
                              >
                                <Link2 className="h-4 w-4" />
                              </button>

                              <button
                                onClick={(e) => toggleFavorite(q.id, e)}
                                className={`icon-btn ${isFavorited ? 'favorited' : ''}`}
                                title={isFavorited ? 'Remove from favorites' : 'Save for later'}
                              >
                                <motion.span
                                  key={isFavorited ? 'fav' : 'unfav'}
                                  initial={reduceMotion || !isMounted ? false : { scale: 1 }}
                                  animate={
                                    reduceMotion || !isMounted
                                      ? {}
                                      : isFavorited
                                        ? { scale: [1, 1.3, 1], transition: { duration: 0.35, ease: EASE_OUT } }
                                        : {}
                                  }
                                  className="flex"
                                >
                                  <Heart className={`h-4 w-4 ${isFavorited ? 'fill-current' : ''}`} />
                                </motion.span>
                              </button>

                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  copyQA(q);
                                }}
                                className="icon-btn"
                                title="Copy question + answer"
                              >
                                <Copy className="h-4 w-4" />
                              </button>

                              <div className="ml-0.5 flex h-8 w-8 items-center justify-center text-ink-low" aria-hidden="true">
                                <ChevronDown className="chevron h-4 w-4" />
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Answer — paper first, ink second */}
                        <AnswerReveal open={isExpanded}>
                          <div className="answer-zone px-5 py-5 sm:px-6 sm:py-6">
                            <div className="answer-overline">Answer</div>
                            <div className="answer-overline-rule" aria-hidden="true" />
                            <div className="prose">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{q.answer}</ReactMarkdown>
                            </div>

                            <div className="mt-7 flex flex-wrap gap-2 border-t border-line-soft pt-5">
                              <button onClick={() => copyQA(q)} className="btn-soft !h-8 !text-xs">
                                <Copy className="h-3.5 w-3.5" /> Copy full answer
                              </button>
                              <button onClick={() => copyQuestion(q)} className="btn-soft !h-8 !text-xs">
                                Copy question only
                              </button>

                              <button onClick={(e) => openFocus(q, e)} className="btn-soft !h-8 !text-xs">
                                <Eye className="h-3.5 w-3.5" /> Read in focus mode
                              </button>

                              <button
                                onClick={(e) => toggleFavorite(q.id, e)}
                                className={`btn-soft !h-8 !text-xs ml-auto ${isFavorited ? '!text-ink-hi' : ''}`}
                              >
                                <Heart className={`h-3.5 w-3.5 ${isFavorited ? 'fill-current text-ember' : ''}`} />
                                {isFavorited ? 'Favorited' : 'Favorite'}
                              </button>
                            </div>
                          </div>
                        </AnswerReveal>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}

            {/* Footer info */}
            <div className="mb-10 mt-16 border-t border-line-soft pt-8 text-center text-sm">
              <p className="mx-auto max-w-md text-ink-low">
                Built for serious AI engineering interview preparation.
              </p>
              <div className="kicker mt-4 flex justify-center gap-3">
                <span>{data.stats.totalQuestions} questions</span>
                <span className="opacity-40">·</span>
                <span>{data.stats.totalSections} sections</span>
                <span className="opacity-40">·</span>
                <span>Saved locally</span>
              </div>
            </div>
          </main>
        </div>
      </div>
      </div>

      {/* Focus / Read Modal — the sanctum */}
      <AnimatePresence>
        {focusQuestion && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="focus-overlay fixed inset-0 z-[100] flex items-center justify-center overflow-y-auto p-4 sm:p-6"
            onClick={closeFocus}
          >
            <motion.div
              initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 28, scale: 0.97 }}
              animate={reduceMotion ? { opacity: 1 } : { opacity: 1, y: 0, scale: 1 }}
              exit={
                reduceMotion
                  ? { opacity: 0 }
                  : { opacity: 0, y: 12, transition: { duration: 0.2, ease: 'easeIn' } }
              }
              transition={reduceMotion ? { duration: 0.2 } : MODAL_SPRING}
              role="dialog"
              aria-modal="true"
              aria-labelledby="focus-question-title"
              className="focus-modal my-auto flex max-h-[min(90vh,calc(100dvh-2rem))] w-full max-w-3xl flex-col p-6 sm:p-10"
              onClick={(e) => e.stopPropagation()}
            >
              <motion.div
                initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 6 }}
                animate={reduceMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
                transition={{ duration: 0.25, ease: EASE_OUT, delay: 0.06 }}
                className="flex min-h-0 flex-1 flex-col"
              >
                <div className="mb-5 flex shrink-0 items-start justify-between gap-4">
                  <div className="min-w-0">
                    {focusSection && (
                      <div className="kicker mb-2">
                        Section {String(focusSection.number).padStart(2, '0')} · {focusSection.title} · ~
                        {focusReadMinutes} min read
                      </div>
                    )}
                    <div id="focus-question-title" className="modal-question break-words pr-2 sm:pr-6">
                      {focusQuestion.question}
                    </div>
                  </div>
                  <button
                    ref={modalCloseRef}
                    onClick={closeFocus}
                    className="icon-btn shrink-0"
                    aria-label="Close focus mode"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>

                <div className="prose min-h-0 flex-1 overflow-y-auto overscroll-contain pr-1">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{focusQuestion.answer}</ReactMarkdown>
                </div>

                <div className="mt-8 flex shrink-0 flex-wrap gap-2 border-t border-line-soft pt-6">
                  <button onClick={() => copyQA(focusQuestion)} className="btn-soft">
                    <Copy className="h-4 w-4" /> Copy full answer
                  </button>
                  <button onClick={() => copyQuestion(focusQuestion)} className="btn-soft">
                    Copy question
                  </button>
                  <button
                    onClick={() => {
                      toggleFavorite(focusQuestion.id);
                    }}
                    className={`btn-soft ml-auto ${favorites.has(focusQuestion.id) ? '!text-ink-hi' : ''}`}
                  >
                    <Heart
                      className={`h-4 w-4 ${favorites.has(focusQuestion.id) ? 'fill-current text-ember' : ''}`}
                    />
                    {favorites.has(focusQuestion.id) ? 'Remove from favorites' : 'Add to favorites'}
                  </button>
                </div>

                <div className="mt-6 text-center text-[11px] text-ink-low">
                  Press <kbd>Esc</kbd> to close · <kbd>C</kbd> to copy answer
                </div>
              </motion.div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
