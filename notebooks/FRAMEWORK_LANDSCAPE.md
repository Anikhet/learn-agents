# Agent framework landscape and selection rules

Researched from official documentation on July 11, 2026. The course remains
LangChain-first. Another framework is included only when it teaches a capability
more directly, exposes a meaningfully different abstraction, or is the reference
SDK for a protocol/platform.

| Framework | Distinctive strength | Course use |
| --- | --- | --- |
| [LangChain](https://docs.langchain.com/oss/python/langchain/overview) | Provider-neutral models, tools, middleware, retrieval, standard agent loop | Primary Python framework |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) | Explicit state, branches, loops, persistence, interrupts, durable execution | Primary orchestration runtime |
| [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) | Planning, context/files, memory, skills, sandboxes, subagents | Long-running/open-ended agents |
| [LangSmith](https://docs.langchain.com/langsmith/observability) | Tracing, datasets, offline/online evaluation, prompts, deployment | Primary lifecycle platform |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | Lightweight runner, handoffs, guardrails, sessions, MCP, hosted tools, realtime voice | Tool-use comparison, handoffs, voice |
| [Google ADK](https://google.github.io/adk-docs/) | Sequential/parallel/loop agents, sessions, evaluation, MCP, reference A2A integration | A2A protocol lesson |
| [Pydantic AI](https://ai.pydantic.dev/) | Typed dependencies, validated outputs, retry-on-validation, testability | Structured-output lesson |
| [LlamaIndex](https://docs.llamaindex.ai/) | Document ingestion, indexes, retrievers, query engines, data-centric agents | RAG lesson |
| [Microsoft Agent Framework](https://learn.microsoft.com/agent-framework/) | Successor path combining AutoGen-style agents with Semantic Kernel enterprise workflows | Landscape comparison; optional Microsoft track |
| [AutoGen](https://microsoft.github.io/autogen/stable/) | Event-driven multi-agent systems and AgentChat teams | Historical/current comparison, not a second core API |
| [CrewAI](https://docs.crewai.com/) | Role/task/crew metaphor plus flows, memory, and knowledge | Comparison only; overlaps Deep Agents/LangGraph |
| [Strands Agents](https://strandsagents.com/) | Model-driven Python agents with AWS integrations and Graph/Swarm/Workflow patterns | Optional AWS track |
| [smolagents](https://huggingface.co/docs/smolagents/) | Minimal code agents and tool-calling agents, local/open model friendliness | Optional local-model/code-agent example |
| [Agno](https://docs.agno.com/) | Pure-Python Agent/Team/Workflow primitives plus AgentOS for sessions, memory, knowledge, evals, tracing, RBAC, and APIs | Production-harness comparison |
| [Vercel AI SDK](https://ai-sdk.dev/) | TypeScript streaming, tool loops, generative UI, React integration | TypeScript/frontend appendix only |
| [Mastra](https://mastra.ai/docs) | TypeScript agents, workflows, memory, observability | TypeScript comparison only |

## Rules for adding a framework

1. Do not replace clear Python with a framework merely to increase import count.
2. Teach one concept with one primary abstraction, then show at most one useful
   comparison.
3. Keep TypeScript frameworks out of Python notebooks; use a labeled appendix.
4. Pin optional framework versions separately from the base course.
5. Use async SDK APIs in Jupyter when the SDK documents an event-loop conflict
   with synchronous wrappers.
6. Every alternative example must explain what capability it adds beyond the
   LangChain implementation.

## Chosen additions

- Tool Use: OpenAI Agents SDK `Agent`, `Runner`, and `function_tool`.
- Multi-Agent Collaboration: OpenAI handoffs, contrasted with agents-as-tools.
- Structured Output: Pydantic AI validated output and typed result.
- RAG: LlamaIndex documents, vector index, and query engine.
- A2A: Google ADK belongs here once a complete local A2A server/client exercise
  is added and tested.
- Voice: OpenAI Realtime Agents belongs here after an audio-capable environment
  is available; the Python SDK uses a live async session rather than a one-shot
  text runner.
- Production harness: Agno demonstrates the path from a Python agent to an
  AgentOS-managed API without replacing the LangGraph durability lesson.

Vanilla Python remains appropriate for algorithms such as scheduling,
statistics, clustering, batching simulation, policy evaluation, and protocol
data-shape demonstrations. Those examples explain mechanics that a framework
would otherwise hide.
