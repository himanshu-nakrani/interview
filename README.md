# AI Engineering Interview Prep

A modern, beautiful Next.js website for the comprehensive AI Engineering Interview Preparation Guide.

**456 questions • 14 sections**

Topics covered:
LLM Fundamentals • Prompt Engineering • RAG • AI Agents • Fine-Tuning • Vector Databases • System Design • LLMOps • Evaluation • Safety • Multimodal • Infrastructure • Coding • Behavioral

## Features

- Full-text live search across all questions and answers
- Clean, dark technical interface
- Expandable answer cards with full Markdown rendering
- One-click copy for questions or full Q&A
- Sidebar table of contents with active section highlighting
- Keyboard shortcut: press `/` to focus search
- Fully responsive (mobile drawer navigation)
- Static site — instant loads

## Getting Started

```bash
npm install
npm run dev
```

Open http://localhost:3000

## Regenerate data (if you edit the source markdown)

The content lives in `content/ai_engineering_interview_prep.md`.

After editing, regenerate the structured data:

```bash
npm run generate
```

## Build

```bash
npm run build
npm start
```

## Project structure

```
content/ai_engineering_interview_prep.md   # Source content
scripts/generate-data.ts                   # Markdown → JSON parser
data/interview-data.json                   # Generated structured data
app/page.tsx                               # The website UI
```

Built with Next.js 16 + Tailwind + React Markdown.

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
