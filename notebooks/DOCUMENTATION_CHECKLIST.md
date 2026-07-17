# LangChain ecosystem teaching checklist

Audited against the official Python documentation on July 11, 2026. A topic is
complete only when both boxes are checked:

- **Taught** means the concept, selection rule, and failure mode are explained.
- **Implemented** means a runnable framework example exists in the named
  notebook. A standard-library simulation does not satisfy this box.

This checklist covers the public tutorial and guide surface relevant to this
course. Provider-specific integration catalogs and API-reference members are
not individual curriculum requirements.

## Foundations and sequencing

- Course environment and product boundaries — `00_course_orientation.ipynb`
  - [x] Taught
  - [x] Implemented: version and credential check
- Provider/model initialization — `01_prompt_chaining.ipynb`
  - [x] Taught
  - [x] Implemented: `init_chat_model("openai:gpt-5.6-sol")`
- Messages and prompt templates — `01_prompt_chaining.ipynb`
  - [x] Taught
  - [x] Implemented: `ChatPromptTemplate`
- Runnable composition — `01_prompt_chaining.ipynb`
  - [x] Taught
  - [x] Implemented: prompt `|` model pipeline
- Token and event streaming — `23_voice_ai_fundamentals.ipynb`
  - [x] Taught
  - [ ] Implemented with LangChain/LangGraph stream modes
- Sync versus async invocation
  - [ ] Taught
  - [ ] Implemented

## LangChain agents

- `create_agent` tool-calling loop — `05_tool_use_function_calling.ipynb`
  - [x] Taught
  - [ ] Implemented in the canonical lesson; later security/observability lessons use it
- Function tools and `@tool` — `05_tool_use_function_calling.ipynb`
  - [x] Taught
  - [ ] Implemented in the canonical lesson
- Typed tool arguments and descriptions — `05_tool_use_function_calling.ipynb`
  - [x] Taught
  - [ ] Implemented with a real agent call
- Tool error handling with `wrap_tool_call` — `12_exception_handling_and_recovery.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Dynamic tool selection through middleware — `39_agent_identity_and_governance.ipynb`
  - [x] Taught
  - [x] Implemented with runtime scope filtering
- Dynamic model selection — `16_resource_aware_optimization.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented with model middleware
- Agent streaming
  - [ ] Taught
  - [ ] Implemented
- Agent state extension through middleware — `08_memory_management.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Runtime context schema — `39_agent_identity_and_governance.ipynb`
  - [x] Taught
  - [x] Implemented

## Structured output and routing

- Model `with_structured_output` — `02_routing.ipynb`
  - [x] Taught
  - [x] Implemented with a Pydantic route
- Agent `response_format` — `24_structured_output_model_agnostic.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Automatic provider/tool strategy selection — `24_structured_output_model_agnostic.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Explicit `ProviderStrategy` — `24_structured_output_model_agnostic.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Explicit `ToolStrategy` and schema-error recovery — `24_structured_output_model_agnostic.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- LangGraph conditional routing — `02_routing.ipynb`
  - [x] Taught
  - [x] Implemented

## LangGraph workflows

- Typed graph state — `02_routing.ipynb`
  - [x] Taught
  - [x] Implemented with `TypedDict`
- Nodes, `START`, `END`, and normal edges — `02_routing.ipynb`
  - [x] Taught
  - [x] Implemented
- Conditional edges — `02_routing.ipynb`
  - [x] Taught
  - [x] Implemented
- Parallel branches and reducers — `03_parallelization.ipynb`
  - [x] Taught
  - [x] Implemented with `Annotated[..., operator.add]`
- Evaluator–optimizer loop — `04_reflection.ipynb`
  - [x] Taught
  - [x] Implemented with a hard attempt limit
- `Command(update=..., goto=...)` — `35_computer_use_browser_agents.ipynb`
  - [x] Taught
  - [x] Implemented
- Dynamic map-reduce with `Send` — `26_mapreduce_summarization_confidence.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Functional API with `@task` and `@entrypoint` — `26_mapreduce_summarization_confidence.ipynb`
  - [ ] Taught
  - [ ] Implemented
- Recursion limits and graceful best-so-far output — `21_exploration_and_discovery.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented with LangGraph
- Retry policy on graph nodes — `12_exception_handling_and_recovery.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Runtime context in graph nodes
  - [ ] Taught
  - [ ] Implemented

## Persistence, memory, and human review

- Thread-level checkpoints — `36_durable_agent_orchestration.ipynb`
  - [x] Taught
  - [x] Implemented with `InMemorySaver`
- Stable `thread_id` configuration — `36_durable_agent_orchestration.ipynb`
  - [x] Taught
  - [x] Implemented
- Production database checkpointer — `36_durable_agent_orchestration.ipynb`
  - [x] Taught
  - [ ] Implemented with SQLite/Postgres
- `interrupt()` and `Command(resume=...)` — `36_durable_agent_orchestration.ipynb`
  - [x] Taught
  - [x] Implemented
- Approve/edit/reject lifecycle — `13_human_in_the_loop.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented with `HumanInTheLoopMiddleware`
- State inspection and history — `36_durable_agent_orchestration.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Time-travel replay and fork — `36_durable_agent_orchestration.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Short-term conversation memory — `08_memory_management.ipynb`
  - [x] Taught
  - [ ] Implemented with a checkpointer
- Cross-thread long-term memory — `08_memory_management.ipynb`
  - [x] Taught
  - [ ] Implemented with a LangGraph store
- User/tenant namespace isolation — `08_memory_management.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented

## Subgraphs and multi-agent systems

- Subagent-as-tool pattern — `07_multi_agent_collaboration.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- LangGraph subgraph with shared state keys — `07_multi_agent_collaboration.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Per-invocation, per-thread, and stateless subgraph persistence — `07_multi_agent_collaboration.ipynb`
  - [ ] Taught
  - [ ] Implemented
- Deep Agents synchronous subagents — `28_production_agent_harness.ipynb`
  - [x] Taught
  - [x] Implemented
- Subagent tool restriction and context isolation — `34_context_engineering.ipynb`
  - [x] Taught
  - [x] Implemented
- Structured subagent responses
  - [ ] Taught
  - [ ] Implemented
- Async subagents and lifecycle controls
  - [ ] Taught as preview
  - [ ] Implemented

## Deep Agents harness

- `create_deep_agent` quickstart — `06_planning.ipynb`
  - [x] Taught
  - [x] Implemented
- Built-in planning and `write_todos` — `06_planning.ipynb`
  - [x] Taught
  - [x] Implemented through the harness
- Context offloading to files — `34_context_engineering.ipynb`
  - [x] Taught
  - [x] Implemented through the harness
- Automatic summarization/compaction — `34_context_engineering.ipynb`
  - [x] Taught
  - [x] Enabled through the harness; explicit configuration example remains
- `StateBackend` — `34_context_engineering.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented explicitly
- `FilesystemBackend` safety boundary — `33_sandboxed_agent_execution.ipynb`
  - [x] Taught
  - [ ] Implemented
- `StoreBackend` durable memory — `08_memory_management.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- `CompositeBackend` routing — `08_memory_management.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Filesystem permissions — `33_sandboxed_agent_execution.ipynb`
  - [x] Taught
  - [x] Implemented
- Sandbox backend and lifecycle/TTL — `33_sandboxed_agent_execution.ipynb`
  - [x] Taught
  - [ ] Implemented against a provider
- Deep Agents `interrupt_on` — `13_human_in_the_loop.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Persistent memory files — `08_memory_management.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Skills and progressive disclosure — `G_appendix_g_coding_agents.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented with a `SKILL.md`
- Skills inheritance and subagent isolation — `G_appendix_g_coding_agents.ipynb`
  - [ ] Taught
  - [ ] Implemented

## Retrieval and RAG

- Document loading and normalization — `14_knowledge_retrieval_rag.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented with LangChain
- Text splitting — `14_knowledge_retrieval_rag.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Embeddings and vector store — `14_knowledge_retrieval_rag.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Retriever invocation and citations — `14_knowledge_retrieval_rag.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Hybrid retrieval and reranking — `29_advanced_production_rag.ipynb`
  - [x] Taught
  - [ ] Implemented with framework integrations
- Retrieval evaluation — `29_advanced_production_rag.ipynb`
  - [x] Taught
  - [ ] Implemented in LangSmith
- RAG tracing — `32_observability_and_tracing.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented as a nested retrieval trace

## LangChain middleware and safety

- Before/after model hooks — `18_guardrails_safety_patterns.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Model-call wrapper — `39_agent_identity_and_governance.ipynb`
  - [x] Taught
  - [x] Implemented
- Tool-call wrapper — `12_exception_handling_and_recovery.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- PII middleware — `18_guardrails_safety_patterns.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Summarization middleware — `34_context_engineering.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented directly
- Model/tool call limits — `16_resource_aware_optimization.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Human approval middleware — `31_agent_security_prompt_injection.ipynb`
  - [x] Taught
  - [x] Implemented
- Prompt-injection trust boundary — `31_agent_security_prompt_injection.ipynb`
  - [x] Taught
  - [x] Implemented with restricted tools and approval

## LangSmith observability

- Automatic LangChain/LangGraph tracing — `32_observability_and_tracing.ipynb`
  - [x] Taught
  - [x] Implemented
- `@traceable` for application spans — `32_observability_and_tracing.ipynb`
  - [x] Taught
  - [x] Implemented
- Trace versus run/span versus thread — `32_observability_and_tracing.ipynb`
  - [x] Taught
  - [x] Demonstrated through nested functions
- Named projects, tags, and metadata — `32_observability_and_tracing.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Sensitive-data redaction and retention — `32_observability_and_tracing.ipynb`
  - [x] Taught
  - [ ] Implemented
- Dashboards and alerts — `32_observability_and_tracing.ipynb`
  - [x] Taught
  - [ ] Implemented/configured
- Feedback and annotation queues — `32_observability_and_tracing.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- OpenTelemetry ingestion — `32_observability_and_tracing.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented

## LangSmith evaluation

- Dataset creation — `19_evaluation_and_monitoring.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented with `Client.create_dataset/create_examples`
- Target function — `37_trajectory_evals_and_statistics.ipynb`
  - [x] Taught
  - [x] Implemented
- Code evaluator — `37_trajectory_evals_and_statistics.ipynb`
  - [x] Taught
  - [x] Implemented
- LLM-as-judge evaluator — `22_evaluation_reliability_deep_dive.ipynb`
  - [x] Taught
  - [ ] Implemented with OpenEvals/LangSmith
- Pairwise evaluator — `22_evaluation_reliability_deep_dive.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Repetitions and concurrency — `37_trajectory_evals_and_statistics.ipynb`
  - [x] Taught
  - [x] Concurrency implemented; repetitions remain
- Experiment comparison — `30_finetuning_routing_prompt_optimization.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Online evaluators and sampling — `32_observability_and_tracing.ipynb`
  - [x] Taught
  - [ ] Configured
- Failing trace to regression dataset loop — `32_observability_and_tracing.ipynb`
  - [x] Taught
  - [ ] Implemented
- Trajectory evaluation — `37_trajectory_evals_and_statistics.ipynb`
  - [x] Taught
  - [x] Implemented with exact tool-path scoring

## LangSmith prompts and deployment

- Push and pull a versioned prompt — `A_appendix_a_advanced_prompting.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented with `Client.push_prompt/pull_prompt`
- Prompt commits and environment tags — `30_finetuning_routing_prompt_optimization.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented
- Playground experiment workflow — `30_finetuning_routing_prompt_optimization.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented as a guided exercise
- LangGraph-compatible application structure — `28_production_agent_harness.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented as deployable source files
- Local Agent Server and Studio — `28_production_agent_harness.ipynb`
  - [ ] Taught
  - [ ] Implemented
- Deployment client invocation — `36_durable_agent_orchestration.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented with `langgraph_sdk`
- Assistants, threads, runs, and cron jobs — `36_durable_agent_orchestration.ipynb`
  - [x] Taught conceptually
  - [ ] Implemented

## Completion gate

- [x] Every currently framework-backed code example has a preceding conceptual lesson.
- [x] Every notebook code cell parses as Python and every notebook parses as JSON.
- [x] Current imports and constructors were smoke-tested against the locked environment.
- [ ] Every concept taught in prose has a runnable framework implementation.
- [ ] Every official tutorial capability selected for this course is taught before first use.
- [ ] All model-backed examples have been executed against a configured provider.
- [ ] All LangSmith examples have been executed against a configured workspace.

## Alternative framework coverage

- OpenAI Agents SDK tool runner — `05_tool_use_function_calling.ipynb`
  - [x] Taught as a comparison to `create_agent`
  - [x] Implemented with async `Runner.run` and `function_tool`
- OpenAI Agents SDK handoffs — `07_multi_agent_collaboration.ipynb`
  - [x] Taught as transfer-of-control rather than agent-as-tool
  - [x] Implemented
- Pydantic AI validated output — `24_structured_output_model_agnostic.ipynb`
  - [x] Taught
  - [x] Implemented with typed `result.output`
- LlamaIndex ingestion/index/query engine — `14_knowledge_retrieval_rag.ipynb`
  - [x] Taught
  - [x] Implemented with source metadata
- Agno Agent/AgentOS path — `28_production_agent_harness.ipynb`
  - [x] Taught
  - [x] Implemented with an AgentOS-ready `Agent`
- Google ADK A2A
  - [x] Framework and protocol fit researched
  - [ ] Implemented as a complete local A2A server/client exercise
- OpenAI Realtime Agents
  - [x] Session lifecycle and notebook constraints researched
  - [ ] Implemented with a real audio transport
- TypeScript Vercel AI SDK/Mastra track
  - [x] Language boundary documented
  - [ ] Implemented in a dedicated TypeScript appendix
