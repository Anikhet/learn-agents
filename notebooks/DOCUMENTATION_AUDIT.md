# Official documentation coverage audit

Audited against the official Python documentation on July 11, 2026. This file
is the source of truth for deciding which framework examples belong in the
course. Product APIs change quickly; recheck the linked pages before upgrading
dependencies or teaching a new constructor.

For per-capability taught/implemented status, use
[DOCUMENTATION_CHECKLIST.md](DOCUMENTATION_CHECKLIST.md). An unchecked
implementation box is a real curriculum gap, not optional polish.

For the researched comparison of OpenAI Agents SDK, Google ADK, Pydantic AI,
LlamaIndex, Microsoft Agent Framework, CrewAI, Strands, smolagents, Agno,
Vercel AI SDK, and Mastra, see
[FRAMEWORK_LANDSCAPE.md](FRAMEWORK_LANDSCAPE.md).

## Framework boundaries

| Product | Official role | Course rule |
| --- | --- | --- |
| [LangChain](https://docs.langchain.com/oss/python/langchain/overview) | High-level agent framework for models, tools, the tool loop, middleware, and integrations | Start here for ordinary model calls and agents. |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) | Low-level orchestration runtime for state, durable execution, streaming, persistence, and HITL | Introduce only when control flow or state must be explicit. |
| [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) | Batteries-included harness for planning, context management, files, memory, sandboxes, and subagents | Use for genuinely open-ended or long-running tasks, not simple chains. |
| [LangSmith](https://docs.langchain.com/langsmith/observability-quickstart) | Platform for traces, evaluation, prompts, and deployment | Trace early; add datasets and evaluators before claiming production readiness. |

## LangChain tutorial surface

| Official topic | Course destination | Status |
| --- | --- | --- |
| Models, messages, prompting, streaming | 01 Prompt Chaining, Appendix A | Framework example added; streaming remains to add. |
| [`create_agent`](https://docs.langchain.com/oss/python/langchain/agents) and tools | 05 Tool Use | Must replace the current simulated dispatcher. |
| [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output) | 02 Routing, 24 Structured Output | Routing uses typed output; provider/tool strategy comparison remains. |
| [Middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview) | 12 Recovery, 16 Optimization, 18 Guardrails, 31 Security, 39 Identity | Security and identity started; retries, fallbacks, PII, and summarization remain. |
| [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) | 13 HITL | Must show approve/edit/reject with a checkpointer and thread ID. |
| [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) | 08 Memory | Must show conversation state through a checkpointer. |
| Retrieval and RAG | 14 RAG, 29 Production RAG | Must replace synthetic retrieval with loaders, splitters, embeddings, and a retriever. |
| Multi-agent patterns | 07 Collaboration | Teach subagent-as-tool before Deep Agents delegation. |
| Agent tests and evals | 19, 22, 37 | LangSmith trajectory work started; unit and integration tests remain. |

## LangGraph tutorial surface

| Official topic | Course destination | Status |
| --- | --- | --- |
| [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) | 02, 03, 04 | Conditional routing, parallel branches/reducers, and bounded loops added. |
| Functional API | 26 Map-reduce | Add `@task`/`@entrypoint` after the graph API is understood. |
| [Workflow patterns](https://docs.langchain.com/oss/python/langgraph/workflows-agents) | 01–06, 26 | Chaining, routing, parallelization, evaluator–optimizer started; orchestrator-worker remains. |
| [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) | 08 Memory, 36 Durable Orchestration | Durable lesson added; database checkpointer remains. |
| [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) | 13 HITL, 35 Browser, 36 Durable | Browser and durable examples added; complete approval lifecycle remains in 13. |
| Time travel and state inspection | 22 Reliability, 36 Durable | Add replay/fork debugging exercise. |
| Streaming | 23 Voice and production notebooks | Add update/message/custom stream modes. |
| Subgraphs and multi-agent | 07 Collaboration, 28 Harness | Add state-schema boundaries and persistence choices. |
| Memory store | 08 Memory, 34 Context | Add cross-thread store and tenant namespaces. |

## Deep Agents tutorial surface

| Official topic | Course destination | Status |
| --- | --- | --- |
| Quickstart and customization | 06 Planning, 28 Harness | Added. |
| [Backends](https://docs.langchain.com/oss/python/deepagents/backends) | 34 Context, 08 Memory | Add State/Store/Composite comparison with correct namespace isolation. |
| [Sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes) | 33 Sandboxing | Permissions added; real provider lifecycle and TTL exercise remain. |
| Permissions | 33 Sandboxing | Added with first-match-wins ordering and scope warning. |
| HITL | 13 HITL, 33 Sandboxing | Add Deep Agents `interrupt_on` comparison. |
| [Synchronous subagents](https://docs.langchain.com/oss/python/deepagents/subagents) | 07, 28, 34 | Added in context/harness lessons; inheritance and structured results remain. |
| [Async subagents](https://docs.langchain.com/oss/python/deepagents/async-subagents) | 03 Parallelization, 28 Harness | Preview only; teach after synchronous delegation and label API stability. |
| [Skills](https://docs.langchain.com/oss/python/deepagents/skills) | Appendix G | Missing; add progressive disclosure and skill isolation. |
| [Memory](https://docs.langchain.com/oss/python/deepagents/memory) | 08 Memory, 34 Context | Missing practical StoreBackend memory with user-scoped namespace. |

## LangSmith tutorial surface

| Official topic | Course destination | Status |
| --- | --- | --- |
| Tracing quickstart and `@traceable` | 32 Observability | Added. |
| Datasets and [evaluation quickstart](https://docs.langchain.com/langsmith/evaluation-quickstart) | 19 Evaluation | Add a small real dataset before advanced statistics. |
| [Offline and online evaluation](https://docs.langchain.com/langsmith/evaluation) | 19, 22, 32, 37 | Started; online sampling and alerts remain. |
| Human, code, LLM-as-judge, and pairwise evaluators | 22 Reliability | Add one calibrated example of each appropriate evaluator type. |
| Prompt playground, prompt versions, and experiments | Appendix A, 30 Optimization | Missing practical prompt version comparison. |
| OpenTelemetry ingestion | 32 Observability | Add after native tracing so trace/span semantics are clear. |
| Deployment and Studio | 28 Harness, 36 Durable | Missing; add only after local graphs, persistence, and tests work. |

## Corrections from this audit

- Use documented `provider:model` identifiers; course examples now use
  `openai:gpt-5.6-sol` rather than the undocumented `openai:gpt-5.6-sol-mini` string.
- Prefer `create_agent` for ordinary tool loops. Do not hand-build a LangGraph
  ReAct loop unless the lesson specifically teaches low-level orchestration.
- Prefer typed structured output for routing and evaluation over parsing prose.
- Every interrupt example needs a checkpointer and stable `thread_id`.
- Deep Agents filesystem permissions cover built-in filesystem tools, not
  arbitrary custom tools, MCP tools, or sandbox shell execution.
- Treat async subagents as preview functionality and label them accordingly.
- Use user- or tenant-scoped namespaces for durable memory; shared writable
  memory is a prompt-injection boundary.
