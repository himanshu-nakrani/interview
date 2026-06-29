import fs from 'fs';
import path from 'path';

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

function parseMarkdown(raw: string): Section[] {
  const lines = raw.split('\n');
  const sections: Section[] = [];
  
  let currentSection: Section | null = null;
  let currentQuestion: Question | null = null;
  let answerLines: string[] = [];
  let questionCounter = 0;

  function flushQuestion() {
    if (currentQuestion && currentSection) {
      currentQuestion.answer = answerLines.join('\n').trim();
      currentSection.questions.push(currentQuestion);
      currentQuestion = null;
      answerLines = [];
    }
  }

  function flushSection() {
    flushQuestion();
    if (currentSection) {
      sections.push(currentSection);
      currentSection = null;
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Skip frontmatter / intro separator early
    if (line.trim() === '---' && sections.length === 0 && !currentSection) {
      continue;
    }

    if (line.startsWith('## ') && /^\##\s+\d+\./.test(line)) {
      flushSection();

      // Extract number and title: "## 1. LLM Fundamentals"
      const match = line.match(/^##\s+(\d+)\.\s+(.+)$/);
      if (match) {
        const number = parseInt(match[1], 10);
        const title = match[2].trim();
        currentSection = {
          id: `section-${number}`,
          number,
          title,
          questions: [],
        };
      }
      continue;
    }

    // Question heading
    if (line.startsWith('### ') && currentSection) {
      flushQuestion();
      const qText = line.replace(/^###\s+/, '').trim();
      questionCounter++;
      currentQuestion = {
        id: `q-${questionCounter}`,
        question: qText,
        answer: '',
      };
      answerLines = [];
      continue;
    }

    // Accumulate answer text (skip empty leading lines for questions)
    if (currentQuestion) {
      // Stop if we hit next heading (handled above)
      answerLines.push(line);
    }
  }

  // Flush last
  flushQuestion();
  flushSection();

  return sections;
}

function main() {
  const mdPath = path.join(process.cwd(), 'content', 'ai_engineering_interview_prep.md');
  const raw = fs.readFileSync(mdPath, 'utf8');

  const sections = parseMarkdown(raw);

  // Calculate stats
  const totalQuestions = sections.reduce((sum, s) => sum + s.questions.length, 0);

  const output = {
    sections,
    stats: {
      totalSections: sections.length,
      totalQuestions,
    },
    generatedAt: new Date().toISOString(),
  };

  const outDir = path.join(process.cwd(), 'data');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  fs.writeFileSync(
    path.join(outDir, 'interview-data.json'),
    JSON.stringify(output, null, 2)
  );

  console.log(`✅ Generated data for ${sections.length} sections, ${totalQuestions} questions.`);
}

main();
