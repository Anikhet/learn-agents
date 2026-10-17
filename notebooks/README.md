# Curriculum map

Every notebook here is generated from a Markdown lesson in [`../lessons/`](../lessons/). Edit the lesson, then run `python generate_tutorial_notebooks.py`. Run `python scripts/verify_notebooks.py` to execute every notebook offline.

Each lesson follows the same arc: **the problem → build it without a framework → how LangGraph does it → what goes wrong in production → what changed by 2026 → interview questions with strong answers → build it yourself**. Offline cells use a scripted model so every notebook runs without API keys. Cells tagged `live` run real models when `RUN_LIVE=1` and `COURSE_MODEL` are set.

## Learning order (not filename order)

### Stage 1: fundamentals
| # | Lesson | Track |
|---|---|---|
| 00 | [What an agent actually is: the loop, from scratch and in LangGraph](08_appendices/00_course_orientation.ipynb) | A |
| 05 | [Tool Use and Function Calling](02_tools_and_protocols/05_tool_use_function_calling.ipynb) | A |
| 24 | [Structured Output, model-agnostic](06_production/24_structured_output_model_agnostic.ipynb) | A |
| 01 | [Prompt Chaining](01_core_patterns/01_prompt_chaining.ipynb) | A |
| 02 | [Routing](01_core_patterns/02_routing.ipynb) | A |
| 03 | [Parallelization](01_core_patterns/03_parallelization.ipynb) | A |
| 17 | [Reasoning Techniques (ReAct, CoT, reasoning models)](01_core_patterns/17_reasoning_techniques.ipynb) | A |
| F | [Under the Hood: what a graph runtime does](08_appendices/F_appendix_f_under_the_hood.ipynb) | A |

### Stage 2: intelligence
| # | Lesson | Track |
|---|---|---|
| 06 | [Planning](01_core_patterns/06_planning.ipynb) | A |
| 34 | [Context Engineering](01_core_patterns/34_context_engineering.ipynb) | A |
| 40 | [Tool Design and Tool Retrieval](02_tools_and_protocols/40_tool_design_and_tool_retrieval.ipynb) | A |
| 41 | [Skills: procedural knowledge on demand](02_tools_and_protocols/41_skills_and_progressive_disclosure.ipynb) | A |
| 14 | [Knowledge Retrieval (RAG)](03_memory_rag_knowledge/14_knowledge_retrieval_rag.ipynb) | C |
| 08 | [Memory Management](03_memory_rag_knowledge/08_memory_management.ipynb) | A |
| 11 | [Goal Setting and Monitoring](06_production/11_goal_setting_and_monitoring.ipynb) | A |
| 04 | [Reflection and verification](01_core_patterns/04_reflection.ipynb) | A |
| 26 | [Map-reduce summarization and confidence](01_core_patterns/26_mapreduce_summarization_confidence.ipynb) | A |

### Stage 3: production engineering
| # | Lesson | Track |
|---|---|---|
| 12 | [Exception Handling and Recovery](06_production/12_exception_handling_and_recovery.ipynb) | A |
| 13 | [Human in the Loop](05_safety_and_oversight/13_human_in_the_loop.ipynb) | A |
| 36 | [Durable Agent Orchestration](06_production/36_durable_agent_orchestration.ipynb) | A |
| 18 | [Guardrails and Safety Patterns](05_safety_and_oversight/18_guardrails_safety_patterns.ipynb) | A |
| 31 | [Agent Security and Prompt Injection](05_safety_and_oversight/31_agent_security_prompt_injection.ipynb) | A |
| 33 | [Sandboxed Agent Execution](05_safety_and_oversight/33_sandboxed_agent_execution.ipynb) | A |
| 16 | [Resource-Aware Optimization](06_production/16_resource_aware_optimization.ipynb) | A |
| 48 | [Decision Models: Jev and typed probabilistic answers](06_production/48_decision_models_jev.ipynb) | A |
| 32 | [Observability and Tracing](04_evals_and_reliability/32_observability_and_tracing.ipynb) | A |
| 19 | [Evaluation and Monitoring](04_evals_and_reliability/19_evaluation_and_monitoring.ipynb) | A |
| 22 | [Evaluation Reliability Deep Dive](04_evals_and_reliability/22_evaluation_reliability_deep_dive.ipynb) | A |
| 37 | [Trajectory Evals and Statistics](04_evals_and_reliability/37_trajectory_evals_and_statistics.ipynb) | A |
| 28 | [Production Agent Harness (Deep Agents)](06_production/28_production_agent_harness.ipynb) | A |
| 39 | [Agent Identity and Governance](05_safety_and_oversight/39_agent_identity_and_governance.ipynb) | A |

### Stage 4: distributed agents
| # | Lesson | Track |
|---|---|---|
| 10 | [Model Context Protocol (MCP)](02_tools_and_protocols/10_model_context_protocol_mcp.ipynb) | A |
| 07 | [Multi-Agent Collaboration](02_tools_and_protocols/07_multi_agent_collaboration.ipynb) | A |
| 15 | [Inter-Agent Communication (A2A)](02_tools_and_protocols/15_inter_agent_communication_a2a.ipynb) | A |
| 35 | [Computer Use and Browser Agents](02_tools_and_protocols/35_computer_use_browser_agents.ipynb) | A |
| 20 | [Prioritization](01_core_patterns/20_prioritization.ipynb) | A |
| 09 | [Learning and Adaptation](04_evals_and_reliability/09_learning_and_adaptation.ipynb) | A |
| 21 | [Exploration and Discovery](01_core_patterns/21_exploration_and_discovery.ipynb) | A |

### Track B: LLM fundamentals
| # | Lesson |
|---|---|
| 42 | [Tokens, embeddings, and the transformer block](09_llm_fundamentals/42_tokens_embeddings_transformer.ipynb) |
| 43 | [Attention, the KV cache, GQA/MLA, and MoE](09_llm_fundamentals/43_attention_kv_cache_gqa_moe.ipynb) |
| 44 | [Decoding, reasoning models, and structured output](09_llm_fundamentals/44_decoding_reasoning_models_structured_output.ipynb) |
| 30 | [Fine-tuning, routing, and prompt optimization](06_production/30_finetuning_routing_prompt_optimization.ipynb) |

### Track C: retrieval and production AI
| # | Lesson |
|---|---|
| 25 | [Embeddings, clustering, themes](03_memory_rag_knowledge/25_embeddings_clustering_themes.ipynb) |
| 45 | [Retrieval internals: BM25, hybrid fusion, rerankers, HNSW, IVF-PQ](03_memory_rag_knowledge/45_retrieval_internals_bm25_hnsw_rerankers.ipynb) |
| 29 | [Advanced Production RAG](03_memory_rag_knowledge/29_advanced_production_rag.ipynb) |

### Track D: infrastructure and interviews
| # | Lesson |
|---|---|
| 38 | [LLM Serving Infrastructure](06_production/38_llm_serving_infrastructure.ipynb) |
| 46 | [AI system design interviews: method and worked problems](10_interviews/46_ai_system_design_interviews.ipynb) |
| 47 | [Interview question bank](10_interviews/47_agent_interview_question_bank.ipynb) |
| 23 | [Voice AI Fundamentals](07_voice/23_voice_ai_fundamentals.ipynb) |
| 27 | [Systems design: a voice platform](07_voice/27_systems_design_voice_platform.ipynb) |

### Capstone and appendices
| # | Lesson |
|---|---|
| D | [Building an Agent: the enterprise support capstone](08_appendices/D_appendix_d_building_an_agent.ipynb) |
| G | [Coding Agents](08_appendices/G_appendix_g_coding_agents.ipynb) |
| E | [AI Agents on the CLI](08_appendices/E_appendix_e_ai_agents_on_the_cli.ipynb) |
| A | [Advanced Prompting](08_appendices/A_appendix_a_advanced_prompting.ipynb) |
| B | [Agentic Interactions](08_appendices/B_appendix_b_agentic_interactions.ipynb) |
| C | [Quick Reference](08_appendices/C_appendix_c_quick_reference.ipynb) |
| H | [Forward Deployed Engineering](08_appendices/H_appendix_h_forward_deployed_engineering.ipynb) |

## Book chapter to lesson

| Book pattern | Lesson | LangGraph primitive |
|---|---|---|
| Prompt Chaining | 01 | node → node → node |
| Routing | 02, 48 | conditional edges, `Command` |
| Parallelization | 03 | `Send` fan-out, reducers |
| Reflection | 04 | generate → critique → revise loop |
| Tool Use | 05, 40 | `bind_tools`, `ToolNode` |
| Planning | 06 | planner → executor → replanner |
| Multi-Agent | 07 | subgraphs, agents as tools, handoffs |
| Memory | 08 | checkpointer + `Store` |
| Learning/Adaptation | 09 | feedback → eval → change |
| MCP | 10 | MCP tools inside the loop |
| Goal Monitoring | 11 | goal in state + termination checks |
| Exception Recovery | 12 | `RetryPolicy`, fallback edges |
| Human-in-the-Loop | 13 | `interrupt()`, `Command(resume=)` |
| RAG | 14, 29, 45 | retrieval node or retrieval tool |
| A2A | 15 | remote agent as a node |
| Resource Optimization | 16 | model routing, pruning, caching |
| Reasoning | 17, 44 | ReAct loop, reasoning models |
| Guardrails | 18, 31 | validation nodes, tool allowlists |
| Evaluation | 19, 22, 37 | traces, datasets, graders |
| Prioritization | 20 | scoring in state, task queues |
| Exploration | 21 | branch → evaluate loops |
| Context engineering (course addition) | 34, 41 | trimming, summarization, offloading, skills |
| Durable execution (course addition) | 36 | checkpoints, durability modes, idempotency |

## Sources

The book *Agentic Design Patterns*; the interview repository (pallavi-shekhar/ai-engineering-interview-questions-company-wise, 35 companies, publicly reported questions); current vendor documentation and engineering posts. Verified research notes with dates and URLs live in [`../docs/research/`](../docs/research/).
