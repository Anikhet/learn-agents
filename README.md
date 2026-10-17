# Agent Engineering Course: Agentic Design Patterns, built in LangGraph

A hands-on course for engineers who want to understand agent architecture, build it, and pass
AI/Agent Engineer interviews. Every lesson follows the same arc:

```text
the problem
  → build it without a framework (plain Python, nothing hidden)
  → how LangGraph does it (and what the primitive is doing underneath)
  → what goes wrong in production
  → what changed by 2026
  → interview questions with strong answers
  → build it yourself
```

Primary stack: **LangGraph** (runtime: state, edges, checkpoints, interrupts, durable execution),
**LangChain** (models, tools, structured output, middleware), **LangSmith** (traces, evals), and
**Deep Agents** (the harness layer, studied after the machinery is understood). OpenAI Agents SDK,
Google ADK, Pydantic AI, MCP and A2A appear where they teach something LangGraph alone does not.

Four tracks run together: **A** agent engineering (the book's 21 patterns plus context
engineering, durable execution, skills, sandboxes, MCP/A2A, evals), **B** LLM fundamentals
(tokens, attention, KV cache, GQA/MoE, decoding, reasoning models), **C** retrieval and
production AI (BM25, hybrid, rerankers, HNSW/IVF-PQ, agentic RAG), **D** infrastructure and
interviews (serving, system design, a question bank).

Start with [the curriculum map](notebooks/README.md).

## Setup

Python 3.11 or newer.

```bash
uv sync
uv run jupyter lab
```

Every notebook runs offline: model calls in offline cells use a scripted stand-in so the control
flow is what you study. Cells tagged `live` run only with `RUN_LIVE=1`, provider credentials such
as `OPENAI_API_KEY`, and `COURSE_MODEL` (for example `openai:gpt-5.6` or
`anthropic:claude-sonnet-5`). Cells tagged `remote` (LangSmith upload) need `RUN_REMOTE=1`,
`LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true`. Keep keys in your shell, never in a notebook.

Optional secondary frameworks: `uv sync --extra alternatives`.

## Working on the course

```bash
python generate_tutorial_notebooks.py          # lessons/**/*.md -> notebooks/**/*.ipynb
python scripts/verify_notebooks.py             # execute every notebook offline, write docs/verification.json
```

Lessons are the source of truth; notebooks are generated. Authoring rules are in
[`lessons/STYLE.md`](lessons/STYLE.md). Verified research notes (dated, with sources) that the
"what changed by 2026" sections draw on are in [`docs/research/`](docs/research/).

## Sources

The book *Agentic Design Patterns*; the interview repository
(pallavi-shekhar/ai-engineering-interview-questions-company-wise: publicly reported questions
from 35 companies across 10 areas); current vendor documentation and engineering posts.
