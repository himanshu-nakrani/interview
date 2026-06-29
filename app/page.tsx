"use client";

import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { Search, Copy, ChevronDown, ChevronUp, X, BookOpen, Menu, Heart, Shuffle, Link2, Eye, Check } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
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

export default function AIInterviewPrep() {
  const [searchTerm, setSearchTerm] = useState('');
  const [activeSection, setActiveSection] = useState<string | null>(null);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [showMobileToc, setShowMobileToc] = useState(false);
  const [favorites, setFavorites] = useState<Set<string>>(() =>
    loadStoredSet('ai-interview-favorites')
  );
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
  const [activeSectionFilter, setActiveSectionFilter] = useState<string | null>(null);
  const [focusQuestion, setFocusQuestion] = useState<Question | null>(null);
  const [viewedQuestions, setViewedQuestions] = useState<Set<string>>(() =>
    loadStoredSet('ai-interview-viewed')
  );
  const headerRef = useRef<HTMLElement>(null);
  const [headerHeight, setHeaderHeight] = useState(112);

  const getScrollOffset = useCallback((extra = 12) => headerHeight + extra, [headerHeight]);

  // Track sticky header height for sidebar offset and scroll targets
  useEffect(() => {
    const header = headerRef.current;
    if (!header) return;

    const update = () => {
      const height = Math.ceil(header.getBoundingClientRect().height);
      setHeaderHeight(height);
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
    localStorage.setItem('ai-interview-favorites', JSON.stringify(Array.from(favorites)));
  }, [favorites]);

  useEffect(() => {
    localStorage.setItem('ai-interview-viewed', JSON.stringify(Array.from(viewedQuestions)));
  }, [viewedQuestions]);

  // Deep link support: open specific question from URL hash
  useEffect(() => {
    const handleHash = () => {
      const hash = window.location.hash.replace('#', '');
      if (hash) {
        // Try to find the question by id
        for (const section of data.sections) {
          const found = section.questions.find(q => q.id === hash);
          if (found) {
            setExpandedIds(prev => new Set(prev).add(hash));
            setTimeout(() => {
              const el = document.getElementById(`card-${hash}`);
              if (el) {
                const top = el.getBoundingClientRect().top + window.scrollY - getScrollOffset();
                window.scrollTo({ top, behavior: 'smooth' });
              }
            }, 80);
            break;
          }
        }
      }
    };

    // Run on mount
    handleHash();

    // Listen for hash changes
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, [getScrollOffset]);

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

  // Close focus modal
  const closeFocus = () => {
    setFocusQuestion(null);
  };

  // Copy Q + A
  const copyQA = (q: Question) => {
    const text = `${q.question}\n\n${q.answer}`;
    navigator.clipboard.writeText(text).then(() => {
      toast.success('Copied to clipboard');
    });
  };

  // Copy just the question
  const copyQuestion = (q: Question) => {
    navigator.clipboard.writeText(q.question).then(() => {
      toast.success('Question copied');
    });
  };

  // Keyboard support for modal
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
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

  // Generate a short preview of the answer for collapsed state (more readable scanning)
  const getAnswerPreview = (answer: string): string => {
    // Take first paragraph or first 180 chars, clean it up
    let preview = answer.split('\n\n')[0].trim();
    preview = preview.replace(/[#*`]/g, '').replace(/\s+/g, ' ');
    if (preview.length > 180) {
      preview = preview.slice(0, 177) + '…';
    }
    return preview;
  };

  // Global search + favorites + section filter
  const filteredSections = useMemo(() => {
    let sections = data.sections;

    // Section filter
    if (activeSectionFilter) {
      sections = sections.filter(s => s.id === activeSectionFilter);
    }

    // Apply search
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

    // Apply favorites filter
    if (showFavoritesOnly) {
      sections = sections
        .map((section) => ({
          ...section,
          questions: section.questions.filter((q) => favorites.has(q.id)),
        }))
        .filter((s) => s.questions.length > 0);
    }

    return sections;
  }, [searchTerm, showFavoritesOnly, favorites, activeSectionFilter]);

  // Auto-expand all matching results when searching
  const searchExpandedIds = useMemo(() => {
    if (!searchTerm.trim()) return null;
    const allMatching = new Set<string>();
    filteredSections.forEach((s) =>
      s.questions.forEach((q) => allMatching.add(q.id))
    );
    return allMatching;
  }, [searchTerm, filteredSections]);

  const effectiveExpandedIds = searchExpandedIds ?? expandedIds;

  // Total visible questions after filter
  const visibleQuestionCount = useMemo(() => {
    return filteredSections.reduce((sum, s) => sum + s.questions.length, 0);
  }, [filteredSections]);

  // Stats for current view
  const expandedCount = useMemo(() => {
    return Array.from(effectiveExpandedIds).filter((id) =>
      filteredSections.some((s) => s.questions.some((q) => q.id === id))
    ).length;
  }, [effectiveExpandedIds, filteredSections]);

  // Toggle a single question
  const toggleQuestion = (id: string) => {
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

  // Open in focus/read mode
  const openFocus = (q: Question, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    markAsViewed(q.id);
    setFocusQuestion(q);
    setExpandedIds(prev => new Set(prev).add(q.id));
  };

  // Shareable link for a question
  const shareQuestion = (q: Question, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    const url = `${window.location.origin}${window.location.pathname}#${q.id}`;
    navigator.clipboard.writeText(url).then(() => {
      // Also update the current URL hash
      window.history.pushState(null, '', `#${q.id}`);
      toast.success('Link copied to clipboard');
    });
  };

  // Toggle favorite
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

  // Expand / collapse all visible
  const expandAll = () => {
    const allIds = new Set<string>();
    filteredSections.forEach((s) =>
      s.questions.forEach((q) => allIds.add(q.id))
    );
    setExpandedIds(allIds);
  };

  const collapseAll = () => {
    setExpandedIds(new Set());
  };

  // Go to random question (great for interview practice)
  const goToRandomQuestion = () => {
    const allQuestions = filteredSections.flatMap(s => s.questions);
    if (allQuestions.length === 0) return;

    const randomQ = allQuestions[Math.floor(Math.random() * allQuestions.length)];
    
    setExpandedIds(prev => new Set(prev).add(randomQ.id));
    markAsViewed(randomQ.id);
    
    setTimeout(() => {
      const el = document.getElementById(`card-${randomQ.id}`);
      if (el) {
        const top = el.getBoundingClientRect().top + window.scrollY - getScrollOffset();
        window.scrollTo({ top, behavior: 'smooth' });
      }
    }, 50);

    toast.success('Random question loaded');
  };

  // Expand or collapse all questions in a specific section
  const toggleSection = (sectionId: string, expand: boolean) => {
    const section = data.sections.find(s => s.id === sectionId);
    if (!section) return;

    setExpandedIds(prev => {
      const next = new Set(prev);
      section.questions.forEach(q => {
        if (expand) {
          next.add(q.id);
          markAsViewed(q.id);
        } else {
          next.delete(q.id);
        }
      });
      return next;
    });
  };

  // Scroll to section
  const scrollToSection = (sectionId: string) => {
    const el = document.getElementById(sectionId);
    if (el) {
      const top = el.getBoundingClientRect().top + window.scrollY - getScrollOffset();
      window.scrollTo({ top, behavior: 'smooth' });
      setActiveSection(sectionId);
      setShowMobileToc(false);
    }
  };

  // Keyboard shortcut: press / to focus search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '/' && document.activeElement?.tagName !== 'INPUT') {
        e.preventDefault();
        const input = document.getElementById('search-input') as HTMLInputElement;
        input?.focus();
      }
      if (e.key === 'Escape' && searchTerm) {
        setSearchTerm('');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [searchTerm]);

  // Track active section on scroll
  useEffect(() => {
    const topMargin = `-${getScrollOffset()}px`;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) {
          setActiveSection(visible.target.id);
        }
      },
      { rootMargin: `${topMargin} 0px -60% 0px`, threshold: [0.2, 0.6] }
    );

    data.sections.forEach((s) => {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, [headerHeight, getScrollOffset]);

  const isSearching = searchTerm.trim().length > 0;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200">
      {/* Top nav / header */}
      <header
        ref={headerRef}
        className="sticky top-0 z-50 border-b border-zinc-800 bg-zinc-950/95 backdrop-blur supports-[backdrop-filter]:bg-zinc-950/80"
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <div className="flex h-16 items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600">
                <BookOpen className="h-5 w-5" />
              </div>
              <div>
                <div className="font-semibold tracking-tight text-xl">AI Engineering</div>
                <div className="text-[10px] text-zinc-500 -mt-1">INTERVIEW PREP</div>
              </div>
            </div>

            <div className="flex items-center gap-3 text-sm">
              <div className="hidden sm:flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900 px-3 py-1 text-xs text-zinc-400">
                <span>{data.stats.totalQuestions} questions</span>
                <span className="text-zinc-700">•</span>
                <span>{data.stats.totalSections} sections</span>
              </div>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="hidden md:inline text-sm text-zinc-400 hover:text-white transition-colors"
              >
                Source
              </a>
            </div>
          </div>
        </div>

        {/* Search bar */}
        <div className="border-t border-zinc-800 bg-zinc-950">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 py-4">
            <div className="flex flex-col sm:flex-row gap-3 items-stretch sm:items-center">
              <div className="relative flex-1">
                <Search className="absolute left-4 top-3.5 h-4 w-4 text-zinc-500" />
                <input
                  id="search-input"
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search questions and answers... (press / to focus)"
                  className="search-input w-full rounded-xl pl-11 pr-10 py-3 text-base placeholder:text-zinc-500 focus:ring-0"
                />
                {searchTerm && (
                  <button
                    onClick={() => setSearchTerm('')}
                    className="absolute right-3 top-3.5 text-zinc-500 hover:text-zinc-300"
                  >
                    <X className="h-4 w-4" />
                  </button>
                )}
              </div>

              <div className="flex flex-wrap items-center justify-end gap-2">
                <button
                  onClick={goToRandomQuestion}
                  className="flex items-center gap-1.5 rounded-xl border border-zinc-800 bg-zinc-900 hover:bg-zinc-800 hover:border-yellow-900/50 px-4 py-3 text-sm font-medium transition-colors action-btn"
                  title="Load a random question (great for practice)"
                >
                  <Shuffle className="h-4 w-4" />
                  <span className="hidden sm:inline">Random</span>
                </button>
                <button
                  onClick={expandAll}
                  className="flex items-center gap-1.5 rounded-xl border border-zinc-800 bg-zinc-900 hover:bg-zinc-800 px-4 py-3 text-sm font-medium transition-colors action-btn"
                >
                  <ChevronDown className="h-4 w-4" /> <span className="hidden sm:inline">Expand</span>
                </button>
                <button
                  onClick={collapseAll}
                  className="flex items-center gap-1.5 rounded-xl border border-zinc-800 bg-zinc-900 hover:bg-zinc-800 px-4 py-3 text-sm font-medium transition-colors action-btn"
                >
                  <ChevronUp className="h-4 w-4" /> <span className="hidden sm:inline">Collapse</span>
                </button>
                <button
                  onClick={() => setShowMobileToc(!showMobileToc)}
                  className="sm:hidden flex items-center gap-2 rounded-xl border border-zinc-800 bg-zinc-900 px-3 py-3 text-sm"
                >
                  <Menu className="h-4 w-4" />
                </button>
              </div>
            </div>

            {/* Results summary + Filters */}
            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
              <div className="flex items-center gap-2">
                {isSearching ? (
                  <span>
                    <span className="text-blue-400 font-medium">{visibleQuestionCount}</span>
                    <span className="text-zinc-500"> result{visibleQuestionCount === 1 ? '' : 's'} for “{searchTerm}”</span>
                  </span>
                ) : showFavoritesOnly ? (
                  <span className="text-rose-400 font-medium">
                    {visibleQuestionCount} favorite{visibleQuestionCount === 1 ? '' : 's'}
                  </span>
                ) : (
                  <span className="text-zinc-500">
                    Showing <span className="text-zinc-300">{visibleQuestionCount}</span> / {data.stats.totalQuestions}
                  </span>
                )}
                
                {expandedCount > 0 && (
                  <span className="stat-pill">
                    {expandedCount} open
                  </span>
                )}
              </div>

              {/* Favorites toggle */}
              <button
                onClick={() => setShowFavoritesOnly(!showFavoritesOnly)}
                className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium border transition-all ${
                  showFavoritesOnly 
                    ? 'bg-rose-500/10 text-rose-400 border-rose-500/30' 
                    : 'bg-zinc-900 border-zinc-800 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-300'
                }`}
              >
                <Heart className={`h-3.5 w-3.5 ${showFavoritesOnly ? 'fill-current' : ''}`} />
                Favorites
                {favorites.size > 0 && (
                  <span className="ml-0.5 px-1.5 rounded bg-zinc-950/60">{favorites.size}</span>
                )}
              </button>

              {(searchTerm || showFavoritesOnly || activeSectionFilter) && (
                <button
                  onClick={() => {
                    setSearchTerm('');
                    setShowFavoritesOnly(false);
                    setActiveSectionFilter(null);
                  }}
                  className="text-xs text-zinc-400 hover:text-white underline"
                >
                  Clear all filters
                </button>
              )}
            </div>

            {/* Quick section filters */}
            {!showFavoritesOnly && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                <button
                  onClick={() => setActiveSectionFilter(null)}
                  className={`filter-pill text-xs px-3 py-1 rounded-full border text-zinc-400 ${
                    !activeSectionFilter 
                      ? 'active' 
                      : 'border-zinc-800 bg-zinc-900'
                  }`}
                >
                  All sections
                </button>
                {data.sections.map((section) => (
                  <button
                    key={section.id}
                    onClick={() => setActiveSectionFilter(section.id === activeSectionFilter ? null : section.id)}
                    className={`filter-pill text-xs px-3 py-1 rounded-full border ${
                      activeSectionFilter === section.id 
                        ? 'active' 
                        : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:text-zinc-300'
                    }`}
                  >
                    {section.number}. {section.title.split(' ').slice(0, 3).join(' ')}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <div className="flex flex-col lg:flex-row gap-8 pt-6 pb-24">
          {/* Sidebar TOC */}
          <aside className="hidden lg:block w-72 shrink-0">
            <div
              className="sidebar-panel sticky overflow-y-auto overscroll-contain"
              style={{
                top: 'var(--site-header-height, 7rem)',
                maxHeight: 'calc(100vh - var(--site-header-height, 7rem) - 1rem)',
              }}
            >
              <div className="mb-3 px-3 text-xs font-semibold uppercase tracking-[1px] text-zinc-500">
                Contents
              </div>
              <nav className="space-y-0.5">
                {data.sections.map((section) => {
                  const originalCount = section.questions.length;
                  const isActive = activeSection === section.id;
                  const isFiltered = activeSectionFilter === section.id;
                  
                  // Count how many would be shown in current filters
                  let displayCount = originalCount;
                  if (showFavoritesOnly) {
                    displayCount = section.questions.filter(q => favorites.has(q.id)).length;
                  }
                  
                  return (
                    <button
                      key={section.id}
                      onClick={() => scrollToSection(section.id)}
                      className={`toc-link flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-sm ${
                        isActive || isFiltered ? 'active text-white bg-zinc-900' : 'text-zinc-400 hover:text-zinc-200'
                      }`}
                    >
                      <span className="truncate pr-3">
                        {section.number}. {section.title}
                      </span>
                      <span className="font-mono text-[10px] text-zinc-600 tabular-nums shrink-0">
                        {displayCount}
                      </span>
                    </button>
                  );
                })}
              </nav>

              <div className="mt-8 px-3 text-[11px] text-zinc-500 leading-relaxed">
                Press <kbd className="rounded bg-zinc-900 px-1.5 py-px font-mono text-[10px]">/</kbd> to search<br />
                <span className="text-zinc-600">Click cards to expand • Heart to favorite</span>
              </div>
            </div>
          </aside>

          {/* Mobile TOC drawer */}
          {showMobileToc && (
            <div className="lg:hidden fixed inset-0 z-[60] bg-black/60" onClick={() => setShowMobileToc(false)}>
              <div
                className="absolute left-0 top-0 bottom-0 w-[min(18rem,85vw)] max-w-full bg-zinc-950 border-r border-zinc-800 p-4 overflow-y-auto overscroll-contain"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex justify-between items-center mb-4">
                  <div className="text-sm font-semibold">Sections</div>
                  <button onClick={() => setShowMobileToc(false)}>
                    <X className="h-5 w-5" />
                  </button>
                </div>
                <div className="space-y-1">
                  {data.sections.map((section) => (
                    <button
                      key={section.id}
                      onClick={() => scrollToSection(section.id)}
                      className="flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-sm hover:bg-zinc-900 text-left text-zinc-300"
                    >
                      <span>
                        {section.number}. {section.title}
                      </span>
                      <span className="text-xs text-zinc-600">{section.questions.length}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Main content */}
          <main className="flex-1 min-w-0">
            {/* Intro banner */}
            {!isSearching && !showFavoritesOnly && !activeSectionFilter && (
              <div className="mb-8 rounded-3xl border border-zinc-800 bg-zinc-900/50 px-6 py-6">
                <div className="max-w-2xl">
                  <p className="text-[15px] leading-relaxed text-zinc-400">
                    A concise reference covering core AI Engineering interview questions across{" "}
                    <span className="text-zinc-300">LLMs, RAG, Agents, Fine-Tuning, Vector DBs, System Design, LLMOps, Evaluation, Safety, Multimodal, Infrastructure, and more.</span>
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2 text-xs">
                    <div className="rounded-full border border-zinc-800 bg-zinc-950 px-3 py-1 text-zinc-400">456 curated questions</div>
                    <div className="rounded-full border border-zinc-800 bg-zinc-950 px-3 py-1 text-zinc-400">Favorites sync in browser</div>
                    <div className="rounded-full border border-zinc-800 bg-zinc-950 px-3 py-1 text-zinc-400">Press <span className="font-mono">/</span> to search</div>
                    <div className="rounded-full border border-zinc-800 bg-zinc-950 px-3 py-1 text-amber-400/90">Try Random for practice</div>
                  </div>
                </div>
              </div>
            )}

            {filteredSections.length === 0 && (
              <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-10 text-center">
                {showFavoritesOnly ? (
                  <>
                    <Heart className="mx-auto h-8 w-8 text-zinc-700 mb-3" />
                    <p className="text-lg">No favorites yet</p>
                    <p className="mt-1 text-sm text-zinc-500">Click the heart icon on any question to save it for later.</p>
                    <button 
                      onClick={() => setShowFavoritesOnly(false)} 
                      className="mt-4 text-sm text-blue-400 hover:text-blue-300"
                    >
                      Browse all questions →
                    </button>
                  </>
                ) : (
                  <>
                    <p className="text-lg">No matches found.</p>
                    <p className="mt-1 text-sm text-zinc-500">Try different keywords or clear your filters.</p>
                  </>
                )}
              </div>
            )}

            {filteredSections.map((section) => (
              <div key={section.id} id={section.id} className="section-header mb-12">
                {/* Section header */}
                <div className="mb-4 flex flex-col gap-3 border-b border-zinc-800 pb-3 sm:flex-row sm:items-end sm:justify-between">
                  <div className="min-w-0">
                    <div className="font-mono text-xs text-blue-500 tracking-[1.5px]">
                      SECTION {section.number}
                    </div>
                    <h2 className="text-xl sm:text-2xl font-semibold tracking-tight break-words">{section.title}</h2>
                  </div>

                  <div className="flex shrink-0 items-center gap-2 self-start sm:self-auto">
                    <div className="text-xs text-zinc-500 font-mono mr-1">
                      {section.questions.length} Q{section.questions.length === 1 ? '' : 's'}
                    </div>
                    <button
                      onClick={() => toggleSection(section.id, true)}
                      className="section-expand-btn"
                    >
                      Expand all
                    </button>
                    <button
                      onClick={() => toggleSection(section.id, false)}
                      className="section-expand-btn"
                    >
                      Collapse
                    </button>
                  </div>
                </div>

                {/* Questions */}
                <div className="space-y-3">
                  {section.questions.map((q, idx) => {
                    const isExpanded = effectiveExpandedIds.has(q.id);
                    const isFavorited = favorites.has(q.id);

                    return (
                      <div
                        id={`card-${q.id}`}
                        key={q.id}
                        className={`question-card group rounded-2xl border overflow-hidden ${isExpanded ? 'expanded' : ''}`}
                      >
                        {/* Question header */}
                        <div
                          role="button"
                          tabIndex={0}
                          aria-expanded={isExpanded}
                          onClick={() => toggleQuestion(q.id)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault();
                              toggleQuestion(q.id);
                            }
                          }}
                          className="question-header w-full px-4 py-4 text-left cursor-pointer sm:px-5 sm:py-[17px]"
                        >
                          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:gap-4">
                            <div className="flex min-w-0 items-start gap-3 sm:contents">
                              <div className="shrink-0 mt-0.5">
                                <div className="w-7 h-7 rounded-md bg-zinc-900 flex items-center justify-center text-[11px] font-mono text-zinc-500 border border-zinc-800">
                                  {idx + 1}
                                </div>
                              </div>

                              <div className="flex-1 min-w-0 pr-1">
                                <div className="question-text group-hover:text-white break-words">
                                  {highlightText(q.question, searchTerm)}
                                </div>

                                {/* Answer preview when collapsed - hugely improves readability/scannability */}
                                {!isExpanded && (
                                  <div className="question-preview">
                                    {getAnswerPreview(q.answer)}
                                  </div>
                                )}

                                {/* Viewed indicator */}
                                {viewedQuestions.has(q.id) && !isExpanded && (
                                  <div className="mt-1 flex items-center gap-1 text-[11px] text-emerald-600/80">
                                    <Check className="h-3 w-3" /> viewed
                                  </div>
                                )}
                              </div>
                            </div>

                            <div className="flex shrink-0 items-center justify-end gap-0.5 border-t border-zinc-800/60 pt-3 sm:border-0 sm:pt-0.5">
                            {/* Focus / Read mode (very interactive) */}
                            <button
                              onClick={(e) => openFocus(q, e)}
                              className="share-btn flex h-8 w-8 items-center justify-center rounded-lg text-zinc-400 hover:bg-zinc-800 hover:text-blue-400 transition-all"
                              title="Open in focus mode for easier reading"
                            >
                              <Eye className="h-4 w-4" />
                            </button>

                            {/* Share link */}
                            <button
                              onClick={(e) => shareQuestion(q, e)}
                              className="share-btn flex h-8 w-8 items-center justify-center rounded-lg text-zinc-400 hover:bg-zinc-800 hover:text-blue-400 transition-all"
                              title="Copy direct link to this question"
                            >
                              <Link2 className="h-4 w-4" />
                            </button>

                            {/* Favorite */}
                            <button
                              onClick={(e) => toggleFavorite(q.id, e)}
                              className={`flex h-8 w-8 items-center justify-center rounded-lg transition-all ${
                                isFavorited 
                                  ? 'text-rose-400 hover:text-rose-500' 
                                  : 'text-zinc-500 hover:text-rose-400 hover:bg-zinc-800/70'
                              }`}
                              title={isFavorited ? "Remove from favorites" : "Save for later"}
                            >
                              <Heart className={`h-4 w-4 ${isFavorited ? 'fill-current' : ''}`} />
                            </button>

                            {/* Copy */}
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                copyQA(q);
                              }}
                              className="flex h-8 w-8 items-center justify-center rounded-lg text-zinc-400 opacity-70 hover:bg-zinc-800 hover:text-white hover:opacity-100 transition-all action-btn"
                              title="Copy question + answer"
                            >
                              <Copy className="h-4 w-4" />
                            </button>

                            {/* Chevron */}
                            <div className="flex h-8 w-8 items-center justify-center text-zinc-500 ml-0.5">
                              {isExpanded ? (
                                <ChevronUp className="h-4 w-4" />
                              ) : (
                                <ChevronDown className="h-4 w-4" />
                              )}
                            </div>
                            </div>
                          </div>
                        </div>

                        {/* Answer - Animated with clearer header */}
                        <AnimatePresence initial={false}>
                          {isExpanded && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              transition={{ duration: 0.2, ease: [0.32, 0.72, 0, 1] }}
                              className="overflow-hidden border-t border-zinc-800 bg-zinc-950/70"
                            >
                              <div className="answer-panel px-5 py-5">
                                <div className="answer-header">
                                  <span>ANSWER</span>
                                </div>
                                <div className="answer-content prose max-w-none">
                                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {q.answer}
                                  </ReactMarkdown>
                                </div>

                                <div className="mt-6 flex flex-wrap gap-2 border-t border-zinc-800 pt-4">
                                  <button
                                    onClick={() => copyQA(q)}
                                    className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-1.5 text-xs font-medium hover:bg-zinc-800 hover:border-zinc-600 active:bg-black action-btn"
                                  >
                                    <Copy className="h-3.5 w-3.5" /> Copy full answer
                                  </button>
                                  <button
                                    onClick={() => copyQuestion(q)}
                                    className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-1.5 text-xs font-medium hover:bg-zinc-800 hover:border-zinc-600 active:bg-black action-btn"
                                  >
                                    Copy question only
                                  </button>

                                  <button
                                    onClick={(e) => openFocus(q, e)}
                                    className="inline-flex items-center gap-2 rounded-lg border border-blue-900/40 bg-blue-950/20 px-4 py-1.5 text-xs font-medium text-blue-300 hover:bg-blue-950/40 hover:text-blue-200 transition-colors"
                                  >
                                    <Eye className="h-3.5 w-3.5" /> Read in focus mode
                                  </button>

                                  <button
                                    onClick={(e) => toggleFavorite(q.id, e)}
                                    className={`ml-auto inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                                      isFavorited 
                                        ? 'border-rose-800 bg-rose-950/40 text-rose-400' 
                                        : 'border-zinc-700 bg-zinc-900 hover:bg-zinc-800 text-zinc-400'
                                    }`}
                                  >
                                    <Heart className={`h-3.5 w-3.5 ${isFavorited ? 'fill-current' : ''}`} />
                                    {isFavorited ? 'Favorited' : 'Favorite'}
                                  </button>
                                </div>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}

            {/* Footer info */}
            <div className="mt-12 mb-10 border-t border-zinc-800 pt-8 text-center text-sm">
              <p className="text-zinc-500 max-w-md mx-auto">
                Built for serious AI engineering interview preparation.
              </p>
              <div className="mt-3 flex justify-center gap-4 text-xs text-zinc-600">
                <span>{data.stats.totalQuestions} questions</span>
                <span>•</span>
                <span>{data.stats.totalSections} sections</span>
                <span>•</span>
                <span>Favorites saved locally</span>
              </div>
            </div>
          </main>
        </div>
      </div>

      {/* Focus / Read Modal - greatly improves readability for long answers */}
      <AnimatePresence>
        {focusQuestion && (
          <div 
            className="fixed inset-0 z-[100] flex items-center justify-center overflow-y-auto bg-black/80 p-4 sm:p-6"
            onClick={closeFocus}
          >
            <motion.div
              initial={{ opacity: 0, y: 30, scale: 0.985 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.985 }}
              transition={{ duration: 0.2, ease: [0.32, 0.72, 0, 1] }}
              className="focus-modal my-auto flex w-full max-w-4xl max-h-[min(90vh,calc(100dvh-2rem))] flex-col rounded-2xl p-5 shadow-2xl sm:p-9"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="mb-5 flex shrink-0 items-start justify-between gap-4">
                <div className="pr-2 text-lg font-medium leading-tight text-zinc-100 break-words sm:pr-6 sm:text-xl">
                  {focusQuestion.question}
                </div>
                <button 
                  onClick={closeFocus} 
                  className="shrink-0 p-1 text-zinc-400 hover:text-white"
                  aria-label="Close focus mode"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <div className="prose max-w-none min-h-0 flex-1 overflow-y-auto overscroll-contain pr-1">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {focusQuestion.answer}
                </ReactMarkdown>
              </div>

              <div className="mt-8 flex shrink-0 flex-wrap gap-2 border-t border-zinc-800 pt-6">
                <button
                  onClick={() => copyQA(focusQuestion)}
                  className="flex items-center gap-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm font-medium transition-colors"
                >
                  <Copy className="h-4 w-4" /> Copy full answer
                </button>
                <button
                  onClick={() => copyQuestion(focusQuestion)}
                  className="flex items-center gap-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 px-4 py-2 text-sm font-medium transition-colors"
                >
                  Copy question
                </button>
                <button
                  onClick={() => {
                    toggleFavorite(focusQuestion.id);
                  }}
                  className={`ml-auto flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${
                    favorites.has(focusQuestion.id) 
                      ? 'border-rose-800 bg-rose-950/30 text-rose-400' 
                      : 'border-zinc-700 hover:bg-zinc-900 text-zinc-300'
                  }`}
                >
                  <Heart className={`h-4 w-4 ${favorites.has(focusQuestion.id) ? 'fill-current' : ''}`} />
                  {favorites.has(focusQuestion.id) ? 'Remove from favorites' : 'Add to favorites'}
                </button>
              </div>

              <div className="text-center text-[10px] text-zinc-600 mt-6">
                Press <kbd className="px-1.5 py-px bg-zinc-900 rounded">Esc</kbd> to close • <kbd className="px-1.5 py-px bg-zinc-900 rounded">C</kbd> to copy answer
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
