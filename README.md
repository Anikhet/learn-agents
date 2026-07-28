# Agentic Design Patterns Tutorial Notebooks

This course teaches agent engineering through one primary stack: LangChain for
agent/tool composition, LangGraph for explicit state and durable workflows,
LangSmith for traces and evaluation, and Deep Agents for long-running agent
harnesses. The gap notebooks include real framework implementations alongside
offline simulations that isolate the underlying algorithm.

## Setup

Python 3.11 or newer is required because current Deep Agents releases no longer
support Python 3.10.

```bash
uv sync
uv run jupyter lab
```

Model-backed cells expect provider credentials such as `OPENAI_API_KEY`.
Tracing and remote evaluation additionally require `LANGSMITH_API_KEY` and
`LANGSMITH_TRACING=true`. Keep keys in your shell or secret manager, never in a
notebook.

The dependency ranges are based on the releases verified on July 11, 2026:
LangChain 1.3.13, LangGraph 1.2.9, LangSmith 0.10.2, and Deep Agents 0.5.9.
See [the folder-based curriculum](notebooks/README.md) for the current map.


