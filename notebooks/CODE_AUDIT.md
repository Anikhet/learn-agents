# Top-to-bottom code audit

The course uses one stack: LangChain for model and tool composition, LangGraph
for explicit workflows, LangSmith for observability and evaluation, and Deep
Agents for open-ended, long-running work. An offline example may accompany the
framework version, but it must not be presented as the production implementation.

## Start here

| Order | Notebook | Practical implementation |
| --- | --- | --- |
| 00 | Course orientation | Verify installed versions and credentials; make no model call. |
| 01 | Prompt chaining | LangChain prompts and model runnables. Do not introduce a graph for a fixed line. |
| 02 | Routing | Typed model decision plus LangGraph conditional edges. |
| 03 | Parallelization | LangGraph parallel branches and an explicit reducer. |
| 04 | Reflection | Bounded LangGraph evaluator–optimizer loop with typed feedback. |
| 06 | Planning | First Deep Agents lesson: planning, todo state, checkpoints, and optional delegation. |
| 17 | Reasoning techniques | Compare strategies through one model interface and evaluate results in LangSmith. |
| 20 | Prioritization | Typed scoring schema followed by deterministic sorting and a budget gate. |
| 21 | Exploration and discovery | LangGraph search loop with a hard experiment budget and best-so-far state. |
| 26 | Map-reduce summarization | Dynamic `Send` fan-out, reducer-based fan-in, and confidence aggregation. |
| 34 | Context engineering | Deep Agents context offloading, memory, compaction, and subagent isolation. |

## Tools and protocols

| Notebook | Practical implementation |
| --- | --- |
| 05 Tool use | `@tool`, `create_agent`, typed arguments, tool errors, and an actual tool call. |
| 07 Multi-agent collaboration | LangChain subagent-as-tool first; Deep Agents delegation for larger tasks. |
| 10 MCP | Connect a real local MCP server and inspect its tools before giving them to an agent. |
| 15 A2A | Validate messages against the current A2A schema and trace handoffs. |
| 35 Computer use | LangGraph observe–propose–policy–act loop with interrupts and a step limit. |

## Memory and knowledge

| Notebook | Practical implementation |
| --- | --- |
| 08 Memory | LangGraph checkpointer for thread memory and a store for cross-thread memory. |
| 14 RAG | LangChain document loading, splitting, embeddings, retriever, and cited answer chain. |
| 25 Embeddings | Real embedding calls plus clustering and inspection of poor clusters. |
| 29 Production RAG | Hybrid retrieval, reranking, metadata filters, and LangSmith retrieval evals. |

## Reliability, safety, and production

| Notebook group | Practical implementation |
| --- | --- |
| 09, 19, 22 | LangSmith datasets, evaluators, repeated runs, and regression gates. |
| 32, 37 | Traces, online/offline evaluation, and trajectory-level scoring. |
| 13, 18 | LangGraph interrupts followed by LangChain guardrail middleware. |
| 31, 33, 39 | Injection boundaries, Deep Agents permissions/sandboxes, and runtime scope filtering. |
| 11, 12, 16 | LangGraph state metrics, retry policies, fallbacks, and budget-aware routing. |
| 24 | LangChain structured output with provider and tool strategies. |
| 28, 36 | Deep Agents harness plus durable LangGraph checkpoints and idempotent effects. |
| 30 | LangSmith experiments comparing prompts/models before considering fine-tuning. |
| 38 | Trace serving latency, token usage, and cache metadata at the model boundary. |

## Specialized and appendix material

Voice notebooks should use real streaming interfaces only after the text-agent
path is stable; their offline timing simulations remain useful. Appendices A–G
should reuse the same APIs rather than create a second framework. Appendix H
uses a LangGraph pilot with measurable rollout gates.
