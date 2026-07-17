# Notebook Topics

The 48 tutorial notebooks are grouped into topic folders. The practical spine is
LangChain + LangGraph + LangSmith + Deep Agents; standard-library simulations
remain only where they make an algorithm easier to inspect offline.

Curriculum maintenance:

- [Official documentation audit](DOCUMENTATION_AUDIT.md)
- [Taught/implemented checklist](DOCUMENTATION_CHECKLIST.md)
- [Top-to-bottom code plan](CODE_AUDIT.md)
- [Agent framework landscape](FRAMEWORK_LANDSCAPE.md)

| Folder | Topic | Notebooks |
| --- | --- | --- |
| `01_core_patterns/` | Foundational patterns plus context engineering | 01, 02, 03, 04, 06, 17, 20, 21, 26, 34 |
| `02_tools_and_protocols/` | Tool use, MCP, A2A, collaboration, and browser agents | 05, 07, 10, 15, 35 |
| `03_memory_rag_knowledge/` | Memory management, RAG, embeddings and clustering, production RAG | 08, 14, 25, 29 |
| `04_evals_and_reliability/` | Evaluation, LangSmith observability, trajectory evals, and statistics | 09, 19, 22, 32, 37 |
| `05_safety_and_oversight/` | Guardrails, HITL, prompt-injection defense, sandboxing, and identity | 13, 18, 31, 33, 39 |
| `06_production/` | Goals, recovery, optimization, structured output, Deep Agents harnesses, durable LangGraph orchestration, and serving | 11, 12, 16, 24, 28, 30, 36, 38 |
| `07_voice/` | Voice AI fundamentals and voice-platform systems design | 23, 27 |
| `08_appendices/` | Course orientation and appendices A–H | 00, A–H |

## Eval coverage assessment (July 2026 audit)

**Covered well** — notebook 22 is the backbone: eval taxonomy (offline/online, unit vs end-to-end), non-determinism and why exact-match fails, LLM-as-judge with bias/calibration (Cohen's kappa), golden/regression datasets, CI gating with tolerance bands. RAG-specific evals (RAGAS, recall@k, MRR) live in 29; evaluator gates in agent loops in 28; regression safety for learned changes in 09 and 30; an end-to-end mini eval harness in Appendix D. The "write the eval before the prompt" discipline is reinforced across most pattern notebooks.

**Gaps** — mostly prose/checklists rather than runnable code:

- No real LLM-as-judge implementation, judge-calibration code, dataset files, or CI eval config in the repo
- Trajectory/agent evals (tool-call matching, step-level graders, multi-turn evals) are named but never demonstrated
- Red-teaming and guardrail evaluation are essentially absent (18 lists guardrails but never evals them)
- No human annotation workflow (labeling rubrics, inter-annotator agreement)
- No statistical rigor: significance testing, confidence intervals, sample-size guidance
- Online monitoring (drift detection, alerting, canary auto-rollback) is asserted in checklists, not shown
- Notebook 19 is a shallow templated overview; 22 carries nearly all the real substance

## Curriculum gaps vs the 2026 hiring market (web research, July 2026)

Ranked by demand in current job descriptions (AI labs, big tech, enterprise) and industry practice:

| Priority | Missing topic | Suggested home |
| --- | --- | --- |
| Critical | Agent security: prompt injection defense, OWASP Agentic Top 10, excessive agency, red teaming | 05_safety_and_oversight |
| Critical | Observability & LLM tracing: OTel spans, Langfuse/LangSmith/Phoenix, eval-score alerting | 04_evals_and_reliability |
| High | Sandboxed code execution: E2B/Modal/microVMs, network egress and credential policy | 05_safety_and_oversight |
| High | Context engineering: attention budgets, compaction, dynamic context assembly | 01_core_patterns |
| High | Computer-use / browser agents: vision-action loops, kill switches (Appendix B surveys this but no hands-on) | 02_tools_and_protocols |
| High | Durable orchestration: LangGraph-style checkpointing, resumable state machines | 06_production |
| High | Trajectory evals, annotation pipelines, statistical rigor for evals | 04_evals_and_reliability |
| High | Forward-deployed engineering: discovery, scoping, enterprise rollout (dominant 2026 lab hiring shape) | 08_appendices |
| Medium | LLM serving infra: vLLM, continuous batching, KV-cache economics | 06_production |
| Medium | Agent identity, auth & governance: non-human identity, OAuth for agents, EU AI Act (enforceable Aug 2026) | 05_safety_and_oversight |
| Low | Agent UX / conversational design; frontier-model selection & pricing fluency | 08_appendices |

Already strong per the same research: core patterns, tool use, MCP/A2A, RAG end-to-end, memory, offline evals, guardrails/HITL, app-layer cost optimization, voice.
