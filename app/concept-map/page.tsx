"use client";

import React, { useLayoutEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  BookOpen,
  Maximize2,
  Minus,
  Network,
  Plus,
  RotateCcw,
  Search,
  Target,
} from "lucide-react";

import rawData from "../../data/interview-data.json";

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

type NodePosition = {
  x: number;
  y: number;
  group: "foundation" | "build" | "operate" | "prove";
  label: string;
};

const data = rawData as Data;

const mapSize = { width: 1120, height: 760 };
const centerNode = { id: "core", x: 560, y: 380 };

const nodePositions: Record<string, NodePosition> = {
  "section-1": { x: 650, y: 88, group: "foundation", label: "Foundations" },
  "section-2": { x: 900, y: 132, group: "build", label: "Prompting" },
  "section-3": { x: 880, y: 240, group: "build", label: "Retrieval" },
  "section-4": { x: 900, y: 430, group: "build", label: "Agents" },
  "section-5": { x: 760, y: 620, group: "build", label: "Adaptation" },
  "section-6": { x: 520, y: 660, group: "foundation", label: "Embeddings" },
  "section-7": { x: 250, y: 610, group: "operate", label: "Design" },
  "section-8": { x: 130, y: 420, group: "operate", label: "Production" },
  "section-9": { x: 170, y: 230, group: "prove", label: "Evaluation" },
  "section-10": { x: 300, y: 126, group: "prove", label: "Safety" },
  "section-11": { x: 1020, y: 330, group: "foundation", label: "Multimodal" },
  "section-12": { x: 90, y: 580, group: "operate", label: "Infra" },
  "section-13": { x: 380, y: 700, group: "build", label: "Coding" },
  "section-14": { x: 100, y: 110, group: "prove", label: "Behavioral" },
};

const sectionSummaries: Record<string, string> = {
  "section-1": "Core model mechanics, token flow, attention, sampling, and inference tradeoffs.",
  "section-2": "Instruction design, structured output, context shaping, and practical prompt patterns.",
  "section-3": "Chunking, retrieval quality, grounding, citations, and document QA architecture.",
  "section-4": "Tool use, planning loops, memory, orchestration, and agent reliability boundaries.",
  "section-5": "SFT, PEFT, LoRA, distillation, evaluation, and adaptation decision points.",
  "section-6": "Embedding models, vector search, ANN indexes, filtering, and retrieval semantics.",
  "section-7": "End-to-end AI product architecture, latency, cost, failure modes, and tradeoffs.",
  "section-8": "Deployment, observability, monitoring, versioning, rollback, and operational quality.",
  "section-9": "Offline and online evals, test sets, human review, regression tracking, and metrics.",
  "section-10": "Risk controls, privacy, policy, prompt injection, misuse, fairness, and governance.",
  "section-11": "Vision, audio, cross-modal retrieval, model choice, and multimodal product patterns.",
  "section-12": "GPU serving, batching, caching, quantization, throughput, and scale planning.",
  "section-13": "Implementation tasks, API integration, debugging, data pipelines, and practical code.",
  "section-14": "Scenario judgment, communication, tradeoff framing, and senior interview signals.",
};

const edges = [
  ["section-1", "section-2"],
  ["section-1", "section-3"],
  ["section-1", "section-5"],
  ["section-1", "section-6"],
  ["section-2", "section-4"],
  ["section-3", "section-6"],
  ["section-3", "section-7"],
  ["section-4", "section-7"],
  ["section-4", "section-8"],
  ["section-5", "section-8"],
  ["section-6", "section-3"],
  ["section-7", "section-8"],
  ["section-8", "section-9"],
  ["section-8", "section-12"],
  ["section-9", "section-10"],
  ["section-10", "section-14"],
  ["section-11", "section-7"],
  ["section-12", "section-8"],
  ["section-13", "section-7"],
  ["section-14", "section-10"],
];

const zoomLevels = [0.82, 1, 1.16];

export default function ConceptMapPage() {
  const [selectedId, setSelectedId] = useState("section-1");
  const [query, setQuery] = useState("");
  const [zoomIndex, setZoomIndex] = useState(1);
  const mapScrollRef = useRef<HTMLDivElement>(null);

  const selectedSection = data.sections.find((section) => section.id === selectedId) ?? data.sections[0];

  const matches = useMemo(() => {
    const term = query.trim().toLowerCase();
    if (!term) return new Set(data.sections.map((section) => section.id));

    return new Set(
      data.sections
        .filter((section) => {
          const haystack = [
            section.title,
            sectionSummaries[section.id],
            ...section.questions.flatMap((question) => [question.question, question.answer]),
          ]
            .join(" ")
            .toLowerCase();

          return haystack.includes(term);
        })
        .map((section) => section.id)
    );
  }, [query]);

  const relatedSections = useMemo(() => {
    const relatedIds = new Set(
      edges
      .filter(([from, to]) => from === selectedId || to === selectedId)
      .map(([from, to]) => (from === selectedId ? to : from))
    );

    return Array.from(relatedIds)
      .map((id) => data.sections.find((section) => section.id === id))
      .filter(Boolean) as Section[];
  }, [selectedId]);

  const totalQuestions = data.stats.totalQuestions;
  const selectedShare = Math.round((selectedSection.questions.length / totalQuestions) * 100);
  const activeEdgeIds = new Set(
    edges
      .filter(([from, to]) => from === selectedId || to === selectedId)
      .map(([from, to]) => `${from}-${to}`)
  );

  useLayoutEffect(() => {
    const scrollElement = mapScrollRef.current;
    const position = nodePositions[selectedId];
    if (!scrollElement || !position) return;

    const centerSelectedNode = () => {
      const zoom = zoomLevels[zoomIndex];
      const viewportWidth = Math.max(scrollElement.clientWidth, Math.min(window.innerWidth - 32, 720));
      const viewportHeight = Math.max(scrollElement.clientHeight, 480);
      const left = position.x * zoom - viewportWidth / 2;
      const top = position.y * zoom - viewportHeight / 2;

      scrollElement.scrollTo({
        left: Math.max(0, left),
        top: Math.max(0, top),
        behavior: "auto",
      });
    };

    centerSelectedNode();
    const frame = requestAnimationFrame(centerSelectedNode);
    const timer = window.setTimeout(centerSelectedNode, 80);

    return () => {
      cancelAnimationFrame(frame);
      window.clearTimeout(timer);
    };
  }, [selectedId, zoomIndex]);

  const setZoom = (direction: "in" | "out" | "reset") => {
    if (direction === "reset") {
      setZoomIndex(1);
      return;
    }
    setZoomIndex((current) => {
      if (direction === "in") return Math.min(zoomLevels.length - 1, current + 1);
      return Math.max(0, current - 1);
    });
  };

  const isFiltering = query.trim().length > 0;

  return (
    <main className="concept-page min-h-screen bg-page text-ink-mid">
      <header className="concept-topbar site-header">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              href="/"
              className="btn-soft !h-10 !w-10 shrink-0 justify-center !px-0"
              aria-label="Back to study guide"
            >
              <ArrowLeft className="h-4 w-4" />
            </Link>
            <div className="brand-tile shrink-0">
              <Network className="h-[18px] w-[18px]" />
            </div>
            <div className="min-w-0">
              <h1 className="brand-name truncate">Visual Concept Map</h1>
              <p className="text-[13px] text-ink-low">
                AI engineering interview topics, dependencies, and practice weight.
              </p>
            </div>
          </div>

          <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <label className="relative min-w-0 sm:w-80">
              <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-low" />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Filter concepts..."
                aria-label="Filter concepts"
                className="search-input w-full !h-10 py-2.5 pl-10 pr-3"
              />
            </label>
            <div className="flex items-center gap-1 rounded-xl border border-line-soft bg-surface-2 p-1">
              <button className="concept-icon-btn" onClick={() => setZoom("out")} aria-label="Zoom out">
                <Minus className="h-4 w-4" />
              </button>
              <button className="concept-icon-btn" onClick={() => setZoom("reset")} aria-label="Reset zoom">
                <RotateCcw className="h-4 w-4" />
              </button>
              <button className="concept-icon-btn" onClick={() => setZoom("in")} aria-label="Zoom in">
                <Plus className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-4 px-4 py-5 sm:px-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <section className="concept-panel overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line-soft bg-surface-2/60 px-4 py-3">
            <div className="flex items-center gap-2 text-sm text-ink-mid">
              <Maximize2 className="h-4 w-4 text-ink-low" />
              <span>
                <span className="font-mono tabular-nums">{matches.size}</span> visible topics
              </span>
            </div>
            <div className="kicker flex items-center gap-2">
              <span>{data.stats.totalSections} concepts</span>
              <span className="h-1 w-1 rounded-full bg-line-strong" />
              <span>{totalQuestions} questions</span>
            </div>
          </div>

          <div className="concept-map-scroll" ref={mapScrollRef}>
            <div
              className="concept-map-sizer"
              style={{
                width: `${mapSize.width * zoomLevels[zoomIndex]}px`,
                height: `${mapSize.height * zoomLevels[zoomIndex]}px`,
              }}
            >
            <div
              className="concept-map-canvas"
              style={{
                width: `${mapSize.width}px`,
                height: `${mapSize.height}px`,
                transform: `scale(${zoomLevels[zoomIndex]})`,
              }}
            >
              <svg className="concept-link-layer" viewBox={`0 0 ${mapSize.width} ${mapSize.height}`} aria-hidden="true">
                {data.sections.map((section) => {
                  const position = nodePositions[section.id];
                  const dimmed = isFiltering && !matches.has(section.id);
                  return (
                    <line
                      key={`core-${section.id}`}
                      x1={centerNode.x}
                      y1={centerNode.y}
                      x2={position.x}
                      y2={position.y}
                      className={
                        selectedId === section.id
                          ? "concept-link concept-link-active"
                          : dimmed
                            ? "concept-link concept-link-dimmed"
                            : "concept-link"
                      }
                    />
                  );
                })}
                {edges.map(([from, to]) => {
                  const fromNode = nodePositions[from];
                  const toNode = nodePositions[to];
                  const id = `${from}-${to}`;
                  const dimmed = isFiltering && (!matches.has(from) || !matches.has(to));
                  return (
                    <line
                      key={id}
                      x1={fromNode.x}
                      y1={fromNode.y}
                      x2={toNode.x}
                      y2={toNode.y}
                      className={
                        activeEdgeIds.has(id)
                          ? "concept-link concept-link-active"
                          : dimmed
                            ? "concept-link concept-link-dimmed"
                            : "concept-link concept-link-secondary"
                      }
                    />
                  );
                })}
              </svg>

              <div className="concept-core-node" style={{ left: centerNode.x, top: centerNode.y }}>
                <Network className="h-6 w-6" />
                <span>AI Engineering</span>
                <small>{totalQuestions} questions</small>
              </div>

              {data.sections.map((section) => {
                const position = nodePositions[section.id];
                const isSelected = section.id === selectedId;
                const isMatched = matches.has(section.id);
                const isRelated = relatedSections.some((related) => related.id === section.id);

                return (
                  <button
                    key={section.id}
                    type="button"
                    onClick={() => setSelectedId(section.id)}
                    className={[
                      "concept-node",
                      `concept-node-${position.group}`,
                      isSelected ? "concept-node-selected" : "",
                      isRelated ? "concept-node-related" : "",
                      !isMatched ? "concept-node-dimmed" : "",
                    ].join(" ")}
                    style={{ left: position.x, top: position.y }}
                    tabIndex={isMatched ? undefined : -1}
                  >
                    <span className="concept-node-kicker">{String(section.number).padStart(2, "0")}</span>
                    <span className="concept-node-title">{section.title}</span>
                    <span className="concept-node-meta">{section.questions.length} questions</span>
                    <span className="concept-node-tag">{position.label}</span>
                  </button>
                );
              })}
            </div>
            </div>
          </div>
        </section>

        <aside className="concept-panel">
          <div className="border-b border-line-soft bg-surface-2/60 px-5 py-4">
            <div className="kicker mb-2 flex items-center gap-2">
              <Target className="h-3.5 w-3.5" />
              Selected concept
            </div>
            <h2 className="section-title">{selectedSection.title}</h2>
          </div>

          <div className="space-y-5 p-5">
            <p className="text-sm leading-6 text-ink-mid">{sectionSummaries[selectedSection.id]}</p>

            <div className="grid grid-cols-3 gap-2">
              <div className="stat-tile">
                <span className="label">Questions</span>
                <span className="value">{selectedSection.questions.length}</span>
              </div>
              <div className="stat-tile">
                <span className="label">Share</span>
                <span className="value">{selectedShare}%</span>
              </div>
              <div className="stat-tile">
                <span className="label">Links</span>
                <span className="value">{relatedSections.length}</span>
              </div>
            </div>

            <div>
              <h3 className="mb-3 flex items-center gap-2 text-sm font-medium text-ink-hi">
                <BookOpen className="h-4 w-4 text-ink-low" />
                Practice entry points
              </h3>
              <div className="space-y-2">
                {selectedSection.questions.slice(0, 4).map((question) => (
                  <Link href={`/#${question.id}`} key={question.id} className="concept-question-link">
                    {question.question}
                  </Link>
                ))}
              </div>
            </div>

            <div>
              <h3 className="mb-3 text-sm font-medium text-ink-hi">Related concepts</h3>
              <div className="flex flex-wrap gap-2">
                {relatedSections.map((section) => (
                  <button
                    key={section.id}
                    type="button"
                    onClick={() => setSelectedId(section.id)}
                    className="rounded-full border border-line-soft bg-surface-2 px-3 py-1.5 text-xs text-ink-mid transition-colors duration-200 hover:border-line-strong hover:text-ink-hi"
                  >
                    {section.title}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </aside>
      </div>
    </main>
  );
}
