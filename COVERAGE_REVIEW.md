# Coverage Review

This audit maps the tutorial notebooks to the major book topics and the core concepts a learner should know.
The notebooks are original companion lessons and intentionally avoid reproducing the book text.

## Coverage Standard

Each topic notebook should include:

- A concise explanation of the pattern.
- Learning objectives.
- Important concepts specific to that pattern.
- Common failure modes.
- Implementation notes.
- A runnable offline example.
- Practice tasks and a design checklist.

## Topic Map

### 00 - Course Orientation

How agentic systems differ from ordinary prompt wrappers, and how to use this notebook series.

Key concepts covered:
- Agentic systems combine goals, context, decisions, actions, feedback, and state.
- The core loop is observe, reason or decide, act, evaluate, and update memory.
- Patterns are reusable control-flow choices, not model features.
- Reliability comes from explicit boundaries: tool contracts, state schemas, evals, and human gates.
- A useful agent should be inspectable before it is autonomous.

### 01 - Prompt Chaining

Break a complex task into ordered model calls where each output becomes the next step's input.

Key concepts covered:
- Decomposition: split a large task into smaller operations with clear inputs and outputs.
- Intermediate artifacts: summaries, extracted fields, plans, critiques, or transformed data passed between steps.
- Validation gates: check each step before downstream work amplifies mistakes.
- Context control: each prompt should receive only the information needed for that step.
- Pipeline design: linear chains are simple, while branching chains need routing and error recovery.

### 02 - Routing

Classify an incoming task and send it to the best specialized handler.

Key concepts covered:
- Intent classification: decide what kind of task arrived before doing the work.
- Specialized handlers: route to prompts, tools, models, agents, or workflows optimized for a case.
- Confidence thresholds: uncertain routes should ask for clarification or use a safe fallback.
- Hierarchical routing: broad category first, narrower route second.
- Evaluation: routing needs confusion-matrix style tests because one wrong route can ruin the task.

### 03 - Parallelization

Run independent subtasks at the same time and merge their results.

Key concepts covered:
- Independence: parallel work is safe only when subtasks do not require each other's outputs.
- Fan-out and fan-in: distribute work, then merge it with a deterministic or review-based reducer.
- Diversity: parallel agents or prompts can explore different perspectives or candidates.
- Resource tradeoff: parallelism lowers latency but can increase cost and coordination complexity.
- Aggregation: merging is often the hardest part and should include conflict handling.

### 04 - Reflection

Use a critic or reviewer loop to improve an initial answer or artifact.

Key concepts covered:
- Producer-critic separation: one component creates, another evaluates.
- Revision loops: critiques should point to actionable changes rather than vague preferences.
- Rubrics: reflection improves when the critic uses explicit criteria.
- Stopping conditions: use max iterations, quality thresholds, or diminishing-return checks.
- Risk: self-critique can reinforce model blind spots unless paired with tests or external evidence.

### 05 - Tool Use Function Calling

Give an agent typed tools and let it select actions instead of only writing text.

Key concepts covered:
- Tool schema: name, description, parameters, return type, and side-effect level form a contract.
- Action selection: the agent chooses when a tool is necessary instead of hallucinating facts.
- Observation passing: tool results become new context for the next decision.
- Safety: tools need allowlists, validation, auth boundaries, and idempotency where possible.
- Error handling: invalid arguments, unavailable services, and partial failures must be represented explicitly.

### 06 - Planning

Create and execute a plan for multi-step goals.

Key concepts covered:
- Task decomposition: convert a goal into ordered, observable steps.
- Plan representation: steps need status, dependencies, owner, and completion criteria.
- Execution monitoring: compare actual observations against the plan.
- Replanning: revise when tools fail, context changes, or a step reveals new constraints.
- Granularity: steps that are too broad are hard to verify; steps that are too small waste overhead.

### 07 - Multi-Agent Collaboration

Coordinate specialized agents that divide work and share results.

Key concepts covered:
- Role specialization: agents should have narrow responsibilities and clear authority.
- Coordination topology: supervisor, network, hierarchy, and custom workflows make different tradeoffs.
- Shared state: agents need a common workspace or message protocol to avoid duplicated or inconsistent work.
- Conflict resolution: disagreements require a policy, adjudicator, or evidence-based merge.
- Observability: multi-agent systems need traces that show who did what and why.

### 08 - Memory Management

Store, retrieve, and summarize useful context across turns or tasks.

Key concepts covered:
- Short-term memory: active task state used inside the current workflow.
- Long-term memory: durable facts, preferences, summaries, and lessons reused later.
- Retrieval: memory is useful only when relevant items can be found at the right time.
- Summarization: old context should be compressed without losing decision-critical details.
- Privacy and decay: memory needs retention rules, deletion, freshness checks, and user control.

### 09 - Learning and Adaptation

Improve future behavior using feedback, examples, and lightweight policy updates.

Key concepts covered:
- Feedback capture: collect ratings, corrections, outcomes, and traces in structured form.
- Adaptation level: update prompts, routing rules, memories, examples, or model weights depending on risk.
- Online vs offline learning: production systems usually favor evaluated offline updates for safety.
- Regression risk: improvements on one task can degrade another without a test suite.
- Self-improvement loop: propose change, test it, compare, promote only if metrics improve.

### 10 - Model Context Protocol MCP

Understand MCP-style tool/resource boundaries and why clean interfaces matter.

Key concepts covered:
- Separation of concerns: MCP-style servers expose tools, resources, and prompts outside the agent.
- Context providers: the agent should request needed context instead of embedding every integration.
- Tool boundaries: each capability has a typed interface and controlled permissions.
- Interoperability: common protocols let different clients use the same capabilities.
- Security: resource access, credentials, and side-effecting tools need strict isolation.

### 11 - Goal Setting and Monitoring

Translate high-level goals into measurable progress signals.

Key concepts covered:
- Goal formulation: convert vague intent into measurable success criteria.
- Monitoring: track progress, blockers, cost, elapsed time, and quality signals.
- Milestones: long-running work needs intermediate checkpoints.
- Stall detection: repeated failed attempts should trigger replanning or escalation.
- Alignment: optimize for the user's real goal, not a proxy metric that is easier to measure.

### 12 - Exception Handling and Recovery

Handle tool failures, invalid outputs, and missing information without collapsing the workflow.

Key concepts covered:
- Failure taxonomy: distinguish user ambiguity, tool errors, invalid outputs, policy blocks, and exhausted budgets.
- Retries: retry transient failures with limits and changed inputs when useful.
- Fallbacks: degrade gracefully to simpler workflows or partial answers.
- Compensation: side-effecting workflows may need rollback or reconciliation steps.
- Escalation: some failures should ask a human rather than continue autonomously.

### 13 - Human in the Loop

Insert human approval, review, or clarification at high-value decision points.

Key concepts covered:
- Control points: add human review where risk, ambiguity, cost, or irreversible action is high.
- Approval payloads: humans need enough context to make a decision quickly.
- Modes: clarify, approve, edit, rank, override, or audit.
- Latency tradeoff: human gates improve safety but slow automation.
- Auditability: record who approved what, when, and with which evidence.

### 14 - Knowledge Retrieval RAG

Ground an agent's answer in retrieved documents or records.

Key concepts covered:
- Retrieval before generation: ground answers in documents, records, or search results.
- Chunking and indexing: retrieval quality depends on document segmentation and metadata.
- Ranking: lexical, vector, hybrid, and reranking approaches trade speed and relevance.
- Grounded synthesis: answers should cite retrieved source ids and avoid unsupported claims.
- Failure modes: stale docs, missing docs, bad chunks, prompt injection in documents, and over-reliance on weak evidence.

### 15 - Inter-Agent Communication A2A

Let agents exchange structured messages instead of implicit prose blobs.

Key concepts covered:
- Message envelope: sender, recipient, intent, payload, correlation id, and timestamp.
- Protocol discipline: agents should exchange structured data rather than ambiguous prose.
- Handoffs: ownership transfer needs task state, constraints, and expected output.
- Conversation state: multi-turn agent-to-agent work needs thread ids and traceability.
- Trust boundaries: not every agent should be allowed to instruct every other agent or tool.

### 16 - Resource-Aware Optimization

Make agents budget-aware across latency, token, cost, and compute constraints.

Key concepts covered:
- Budget dimensions: tokens, latency, money, tool quotas, memory, and human attention.
- Strategy selection: choose the cheapest workflow that can meet the quality target.
- Early stopping: stop when expected benefit falls below expected cost.
- Caching: reuse deterministic or stable results to reduce repeated work.
- Graceful degradation: return a partial result or lower-cost method when budgets are constrained.

### 17 - Reasoning Techniques

Use explicit reasoning scaffolds such as decomposition, verification, and comparison.

Key concepts covered:
- Decomposition: break the problem into explicit subquestions or constraints.
- Verification: check conclusions against facts, requirements, and edge cases.
- Comparison: evaluate alternatives with criteria rather than preference alone.
- Assumption tracking: state what is known, inferred, missing, or uncertain.
- Structured outputs: reasoning is easier to evaluate when represented as tables, plans, or checklists.

### 18 - Guardrails Safety Patterns

Constrain inputs, outputs, and tool actions to reduce unsafe or off-policy behavior.

Key concepts covered:
- Input guardrails: detect unsafe, irrelevant, malicious, or policy-violating requests.
- Tool guardrails: restrict dangerous tools, validate arguments, and require approval for side effects.
- Output guardrails: check for unsupported claims, sensitive data, unsafe advice, or format violations.
- Prompt-injection resistance: treat retrieved or user-provided text as data, not instructions.
- Defense in depth: combine policy, schemas, sandboxing, evals, logging, and human review.

### 19 - Evaluation and Monitoring

Measure agent quality with tests, traces, and operational metrics.

Key concepts covered:
- Task evals: fixed cases with expected behavior for routes, tools, answers, and workflows.
- Reference-free checks: rubric scoring, constraint checks, and human review where exact answers are unavailable.
- Operational monitoring: latency, cost, tool errors, retries, escalation rate, and user corrections.
- Trace inspection: debugging agents requires step-level inputs, outputs, decisions, and observations.
- Continuous improvement: eval failures should become new regression tests.

### 20 - Prioritization

Rank tasks, memories, tool calls, or agent work by value and urgency.

Key concepts covered:
- Scoring: rank work by urgency, value, risk, dependencies, and cost.
- Queues: agents often need task queues, memory queues, or tool-call queues.
- Preemption: urgent or high-risk tasks may interrupt lower-priority work.
- Fairness: pure priority can starve important but non-urgent tasks.
- Dynamic updates: priorities should change as new information arrives.

### 21 - Exploration and Discovery

Let agents search a solution space while keeping experiments bounded and observable.

Key concepts covered:
- Hypothesis generation: propose multiple candidate solutions or search paths.
- Bounded experimentation: set budget, stop criteria, and evaluation metrics before exploring.
- Exploration vs exploitation: balance trying new paths with using the best-known path.
- Search logs: record candidates, evidence, decisions, and discarded options.
- Discovery risk: unconstrained exploration can drift from the goal or burn resources.

### A - Appendix A Advanced Prompting

Reusable prompting tactics that support agentic workflows.

Key concepts covered:
- Instruction hierarchy: separate role, task, context, constraints, examples, and output format.
- Few-shot examples: examples steer behavior but can overconstrain if too narrow.
- Context engineering: select, compress, and package only relevant information.
- Structured outputs: schemas make downstream parsing and validation easier.
- Prompt optimization: test variants against evals rather than relying on intuition.

### B - Appendix B Agentic Interactions

Common interaction models between users, agents, tools, and other agents.

Key concepts covered:
- Interaction topology: user-agent, agent-tool, agent-agent, supervisor, and hierarchical flows.
- Sync vs async: short tasks can block; long tasks need status, checkpoints, and resumability.
- Turn-taking: define who can speak, act, ask, or approve at each stage.
- State ownership: decide where task state lives and who may mutate it.
- User experience: agent interactions should make progress and uncertainty visible.

### C - Appendix C Quick Reference

A compact reference notebook for pattern selection.

Key concepts covered:
- Pattern selection: start from the failure mode or system need, not from the pattern name.
- Composability: real systems often combine routing, tools, memory, guardrails, and evals.
- Tradeoffs: every pattern adds overhead and should earn its place.
- Inspection: prefer designs where you can see inputs, decisions, outputs, and costs.
- Minimum viable agent: begin with deterministic control flow, then add model calls.

### D - Appendix D Building an Agent

Assemble multiple patterns into a small complete agent.

Key concepts covered:
- Architecture: combine router, planner, tools, memory, guardrails, and evaluator.
- State model: define a single task state object that every step updates.
- Trace model: persist step-by-step decisions and observations.
- Testing: create focused evals for each component and an end-to-end smoke test.
- Deployment: start with constrained autonomy and add permissions gradually.

### E - Appendix E AI Agents on the CLI

Design command-line agent workflows with clear inputs, logs, and exit behavior.

Key concepts covered:
- Command surface: clear arguments, flags, stdin/stdout behavior, and exit codes.
- Machine-readable mode: JSON output helps compose CLI agents with other tools.
- Progress logs: stderr is useful for human-readable status while stdout remains parseable.
- Non-interactive operation: support automation without hidden prompts.
- Failure behavior: return useful errors and nonzero exit codes.

### F - Appendix F Under the Hood

Inspect the internal moving parts of an agent runtime.

Key concepts covered:
- Runtime loop: receive task, construct context, call model, dispatch tools, update state, repeat.
- Schedulers: decide which task or agent step runs next.
- Context windows: manage what enters the model at each call.
- Tracing: every model call, tool call, and state mutation should be observable.
- Determinism boundaries: separate deterministic orchestration from probabilistic model outputs.

### G - Appendix G Coding Agents

Apply agentic patterns to software engineering tasks.

Key concepts covered:
- Repo orientation: inspect structure, tests, and conventions before editing.
- Patch planning: keep changes scoped and traceable to the bug or feature.
- Tool use: search, read, edit, run tests, and inspect diffs in a tight loop.
- Verification: tests, linters, type checks, and manual review are feedback signals.
- Risk control: avoid unrelated refactors and destructive git operations.

