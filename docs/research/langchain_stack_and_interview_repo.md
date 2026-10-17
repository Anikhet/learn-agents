# Verified: LangChain stack, Deep Agents 0.7, ADK, Pydantic AI, interview repository (checked 2026-09-27)

## Versions (PyPI, 2026-09-27)

| Package | Course pin (installed) | Latest | Notes |
|---|---|---|---|
| langchain | 1.3.13 | 1.4.2 (2026-09-18) | |
| langgraph | 1.2.9 | 1.2.12 (2026-09-21) | |
| langsmith | 0.10.2 | 0.14.1 (2026-09-25) | |
| deepagents | 0.5.9 | 0.7.19 (2026-09-24) | 0.7.0 GA 2026-07-29; requires langchain>=1.4.2, langchain-core>=1.6.4 |
| google-adk | (optional) | 2.10.0 | docs now at adk.dev |
| pydantic-ai | (optional) | 2.51.0 | docs now at pydantic.dev/docs/ai |
| openai-agents | 0.18.2 | 0.22.3 | |

## How LangChain positions the layers (docs.langchain.com/oss/python/concepts/products)
- LangGraph = agent runtime: "low-level orchestration framework and runtime for building, managing, and deploying long-running, stateful agents"; durable execution, streaming, human oversight; compared to Temporal/Inngest.
- LangChain = agent framework: content blocks, the agent loop, middleware; `create_agent` is "a minimal, highly configurable agent harness"; "Agent = Model + Harness".
- Deep Agents SDK = agent harness (Claude Agent SDK in the same category): "the same core tool calling loop as other agent frameworks, but with built-in capabilities that make agents reliable for real tasks": planning, file systems for context management, subagents.

## Deep Agents 0.7 (CONFIRMED)
- 0.7.0 headline: "A leaner, more configurable harness by default." Default-turn input tokens down 65% (5,395 -> 1,895); tool-description tokens down 43%.
- Breaking: `TodoListMiddleware`/`write_todos` now OPT-IN; base prompt empty (`BASE_AGENT_PROMPT` deprecated); `FilesystemBackend`/`LocalShellBackend` default `virtual_mode=True`; `write_file` overwrites; recursive `delete` tool; `StoreBackend` requires explicit `namespace`.
- `create_deep_agent(model, tools, system_prompt, subagents, backend, skills, memory, middleware, interrupt_on, checkpointer, store, response_format, state_schema, context_schema)`.
- Subagents: built-in `task` tool; dict specs (name, description, system_prompt, tools, model, middleware); `CompiledSubAgent` wraps a compiled graph; parallel `task()` calls; subagent returns one final report.
- Backends: StateBackend, FilesystemBackend, LocalShellBackend, StoreBackend, CompositeBackend (route `/memories/` to the store). Tools: ls, read_file, write_file, edit_file, delete, glob, grep, execute.
- Memory: `memory=` paths to AGENTS.md-style files; cross-thread via StoreBackend under `/memories/`; namespaces by assistant_id or user_id. Short-term = thread checkpoints; long-term = store.
- Context engineering: `SummarizationMiddleware` at 85% of max_input_tokens (fallback 170k trigger / keep 6 messages), keeps ~10% recent; tool results > 20,000 tokens offloaded to a file with a 10-line preview; oversized old tool inputs truncated.
- Sandboxes: `BaseSandbox.execute()`; LangSmithSandbox, DaytonaSandbox, E2BSandbox, ModalSandbox, RunloopSandbox, VercelSandbox; QuickJS `CodeInterpreterMiddleware`.
- Skills: `skills=["./dir/"]`, SKILL.md with YAML frontmatter (name, description); three-layer progressive disclosure (metadata at startup -> body on activation -> scripts/assets on demand); default `/skills/`.
- HITL: `interrupt_on={"tool": True | InterruptOnConfig(allowed_decisions=[...])}`; decisions approve/edit/reject/respond; resume `Command(resume={"decisions": [...]})`; checkpointer REQUIRED.
- Prompt caching of static system-prompt sections for Anthropic/Bedrock.

## langchain 1.x `create_agent`
Params: model (string "provider:model" or instance), tools, system_prompt, middleware, response_format (Pydantic -> `structured_response`), checkpointer, store, state_schema, context_schema, name.
Built-in middleware: ToolErrorMiddleware, ToolRetryMiddleware, ModelRetryMiddleware, ModelFallbackMiddleware, SummarizationMiddleware, HumanInTheLoopMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware, PIIMiddleware, TodoListMiddleware, LLMToolSelectorMiddleware, ProviderToolSearchMiddleware, ShellToolMiddleware, FilesystemMiddleware, SubAgentMiddleware, RubricMiddleware, FilesystemFileSearchMiddleware, ContextEditingMiddleware, LLMToolEmulator.

## LangGraph 1.x primitives
- Durability modes: "sync" (persist before next step), "async" (persist while next step runs), "exit" (persist only at graph exit).
- Checkpointers: InMemorySaver, SqliteSaver, PostgresSaver/AsyncPostgresSaver, MongoDB, Redis, Oracle. thread_id in config.
- interrupt(value) + Command(resume=...); requires checkpointer + thread_id; on resume the node RE-RUNS FROM THE TOP, so pre-interrupt side effects must be idempotent. Command(update, goto, graph=Command.PARENT, resume).
- Send("node", state) from conditional edges for dynamic fan-out (map-reduce).
- Store: InMemoryStore, PostgresStore; tuple namespaces; put/get/search; semantic search via index={"embed", "dims"}; `runtime.store` in nodes.
- Subgraphs as nodes; Command.PARENT; streaming subgraphs=True.
- RetryPolicy(max_attempts, initial_interval, backoff_factor, max_interval, jitter, retry_on); default retry_on is conservative (connection errors, 5xx). LangChain blog "Fault Tolerance in LangGraph" (2026-06-04) adds TimeoutPolicy, per-node error_handler, set_node_defaults (exact version UNCONFIRMED).
- CachePolicy(key_func, ttl) per node.
- Streaming modes: values, updates, messages, custom (get_stream_writer), checkpoints, tasks, debug.
- Time travel: get_state_history -> invoke with checkpoint_id replays (LLM calls and interrupts fire again); update_state forks a new branch.
- recursion_limit default 1000 super-steps (note: not 25).

## Google ADK 2.x
SequentialAgent, ParallelAgent, LoopAgent; LLM-driven transfer; agent-as-tool; 2.0 graph workflows (Workflow, nodes, edges, StringRoute/IntRoute/BoolRoute); A2A expose/consume (experimental); McpToolset (stdio, Streamable HTTP, SSE; tool_filter); eval with tool_trajectory_avg_score, response_match_score, rubric metrics via adk eval / pytest; deploy to Agent Runtime, Cloud Run, GKE.

## Pydantic AI 2.x
output_type with ToolOutput/NativeOutput/PromptedOutput/TextOutput; @agent.output_validator + ModelRetry; durable execution with Temporal, DBOS, Prefect, Restate, AWS Lambda (TemporalAgent wrapper path deprecated for v3); MCP(...) capability built on FastMCP (tools, resources, elicitation, OAuth); pydantic-graph (BaseNode, Graph, End) with the caveat "don't use a nail gun unless you need a nail gun".

## The interview repository (CONFIRMED)
https://github.com/pallavi-shekhar/ai-engineering-interview-questions-company-wise — "Real interview questions asked in AI Engineering interviews at 35 companies", "compiled from publicly reported interview experiences", 10 areas, single README. Company tags are reported signals, not guarantees.

### Agents and Tool Use (verbatim)
- Explain the ReAct pattern and what it solves over chain-of-thought alone.
- How do you handle tool-call errors, timeouts and retries in an agentic loop? (OpenAI, Cognition)
- What is the difference between structured output and function calling? (Mistral AI, Apple)
- What is MCP (Model Context Protocol) and how does it differ from traditional function calling? (Microsoft)
- How many tools is too many? How do you design tool schemas an LLM can actually use correctly? (Anthropic, Cognition)
- How does multi-agent orchestration work, and when does it break down? (Cognition)
- Design memory for a long-running agent: what do you store, where, and how do you retrieve it? (Anthropic)
- How does an agent decide when to call a tool versus answer from its own knowledge?
- What makes an agent loop terminate correctly? How do you bound cost and steps?
- How do you make an agent's actions reversible, or at least auditable, in a production system? (Palantir)
- Design human-in-the-loop approval for an agent that takes consequential actions. (OpenAI, Palantir)
- Your agent drifts after a long run and confidently works on the wrong thing. Diagnose it. (Cognition)

### RAG and Retrieval
- What chunking strategy would you use for a large technical documentation corpus, and why? (Glean)
- How do you choose between a sparse retriever (BM25) and a dense retriever? When do you need both? (Microsoft, Perplexity, Glean)
- What is a reranker, when should you use one, and what does a cross-encoder cost you? (Cohere, Microsoft, Perplexity)
- How would you evaluate the quality of a RAG pipeline: retrieval and generation separately? (Cohere)
- What is HyDE and when does it outperform standard dense retrieval?
- How does agentic RAG differ from standard RAG, and when is the extra complexity justified?
- What causes semantic drift in embedding search and how do you detect it? (Cohere)
- Design permission-aware retrieval: users must never see content they can't access in the source system. (Microsoft, Databricks, Glean, Palantir)
- Compare HNSW, IVF-PQ and flat indexes. How do you pick, and what does recall@k cost in latency?
- How do you handle tables, figures and multi-column PDFs in a retrieval pipeline?
- How do you keep an index fresh when the underlying corpus changes continuously? (Perplexity, Cursor)
- How do you attribute every claim in a generated answer to a specific retrieved span? (Perplexity, Harvey, Abridge)

### Evaluation and Observability
- Design an LLM-as-judge evaluation. What are its known biases and how do you correct for them? (Perplexity)
- How do you build an eval set when there is no labelled ground truth and experts are expensive? (Cohere, Harvey)
- How do you detect and measure hallucinations in a production RAG system? (Anthropic, OpenAI, Cursor)
- Design the regression gate that decides whether a prompt or model change ships. (Anthropic)
- Why do benchmark scores improve while users say the system got worse? Enumerate the reasons. (Cognition)
- What is benchmark contamination and how do you guard against it? (Zhipu AI, Alibaba, Scale AI)
- What observability does a production LLM system need: traces, spans, costs, feedback?
- How do you manage prompt versioning and rollbacks in production?
- Design online evaluation: what do you log, what do you sample, and what do you A/B? (Perplexity)
- How would you evaluate an agent, as opposed to a single model response? (Moonshot AI, Zhipu AI, Scale AI, Cognition)

### LLM Internals and Architecture
- Explain scaled dot-product attention and why the 1/sqrt(d_k) scaling factor matters.
- What is the KV cache, and what are its memory implications at scale? Derive the formula. (OpenAI, xAI, Mistral AI, Amazon, Apple, NVIDIA, Together AI, Character.AI)
- What are MQA and GQA, and what do they trade away? (Meta, Mistral AI)
- What is Multi-head Latent Attention (MLA) and why did DeepSeek introduce it? (DeepSeek, Moonshot AI)
- Explain FlashAttention. It does not reduce FLOPs, so why is it faster? (Together AI)
- How does Byte Pair Encoding work, and what are its failure modes (numbers, code, non-Latin scripts)? (Alibaba, Sarvam AI, Hugging Face)
- What is positional encoding in transformers, and how has it evolved (sinusoidal, learned, RoPE, ALiBi)?
- Explain RoPE and how position interpolation / YaRN extend context beyond the trained length. (Meta, Moonshot AI, Alibaba)
- What do the Chinchilla scaling laws say, and how do they differ from earlier scaling intuitions? (Anthropic)
- What is a mixture-of-experts architecture and how does it scale capacity without scaling FLOPs? (Mistral AI, Cohere, DeepSeek, Moonshot AI, Zhipu AI, Alibaba)
- Explain the difference between pre-training, supervised fine-tuning and preference optimisation. (Meta, Scale AI)
- Compare greedy, beam search, top-k, top-p and temperature sampling. When does each fail? (Google DeepMind, Apple, Perplexity)
- What is the lost-in-the-middle problem in long contexts and how do you address it? (Moonshot AI)
- Why is LayerNorm placed pre-block in modern transformers, and what is RMSNorm?
- Explain SwiGLU and why gated activations replaced ReLU/GELU in modern LLM MLP blocks. (Meta)
- Walk me through what happens, tensor by tensor, in one forward pass of a decoder-only transformer. (Anthropic)

### AI System Design
- Design an enterprise RAG assistant over 10M documents with per-user permissions. (OpenAI, Microsoft, Amazon, Databricks, Scale AI)
- Design a code assistant: repo indexing, context assembly, edit application, evaluation. (Cursor)
- Design a customer-support agent that can take real actions, with escalation to humans. (Sierra, consumer-scale ML companies)
- Design semantic search over a large product catalogue. (Character.AI)
- Design a content-moderation system combining classifiers and LLMs. (Meta)
- Design a document-intelligence pipeline: scanned PDFs in, structured fields out, at 10M documents. (Palantir)
- Design a Text-to-SQL system over a warehouse with thousands of tables. (Databricks, Palantir)
- Design a meeting assistant: recording, diarisation, summary, action items, integrations. (Microsoft)
- Design an LLM gateway: routing across providers, failover, caching, budgets and rate limits. (Perplexity, Palantir)
- Design the serving stack for a consumer chat assistant at hundreds of millions of users. (Anthropic, OpenAI, Google DeepMind, Meta, xAI)

Sibling repos: amitshekhariitbhu/ai-engineering-interview-questions (topic-wise); ombharatiya/AI-Engineer-Interview-Questions (33 companies, worked answers).
