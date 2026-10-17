# Verified sources: protocols, vendor guidance, harnesses (checked 2026-09-27)

Legend: CONFIRMED = verified on a primary source; CORRECTED = a common claim that is wrong or imprecise; UNCONFIRMED = not verifiable on a primary source. Lesson authors: cite only CONFIRMED items as fact.

## Model Context Protocol (MCP)

- CONFIRMED: latest spec revision is `2026-07-28` (previous `2025-11-25`). Changelog: https://modelcontextprotocol.io/specification/2026-07-28/changelog . Blog: https://blog.modelcontextprotocol.io/posts/2026-07-28/
- CONFIRMED: stateless protocol core. `initialize` handshake and `Mcp-Session-Id` removed; every request carries `_meta.io.modelcontextprotocol/protocolVersion` and client capabilities; new mandatory `server/discover` RPC. `ping`, `logging/setLevel`, SSE resumability removed.
- CONFIRMED: load-balancer friendly. Servers scale behind round-robin with no sticky sessions. New required headers `Mcp-Method`, `Mcp-Name` on Streamable HTTP POSTs let gateways route/authorize without parsing bodies. Change notifications via opt-in `subscriptions/listen` POST stream.
- CONFIRMED: cacheable listings. `tools/list`, `prompts/list`, `resources/list`, `resources/read` return `ttlMs` and `cacheScope` (public/private). Servers SHOULD return tools in deterministic order (prompt-cache hits).
- CONFIRMED: authorization hardened. RFC 9207 `iss` validation; credentials bound to issuing authorization server; OAuth Dynamic Client Registration deprecated in favor of Client ID Metadata Documents (CIMD).
- CONFIRMED: extension system (`extensions` in capabilities). Official extensions: Tasks, MCP Apps (server-rendered UIs), Enterprise Managed Authorization. https://modelcontextprotocol.io/docs/extensions/overview
- CORRECTED: Tasks are an official EXTENSION, not core. Polling via `tasks/get`, `tasks/update` for client input, servers may return task handles unsolicited. https://modelcontextprotocol.io/extensions/tasks/overview
- CONFIRMED: Multi Round-Trip Requests (MRTR) replace server-initiated requests. Results carry `resultType` ("complete" | "input_required"); server returns `inputRequests` (elicitation), client retries with `inputResponses`.
- CONFIRMED: Roots, Sampling, Logging DEPRECATED (12-month window). Migrations: pass paths via tool params; call LLM APIs directly instead of Sampling; log to stderr/OpenTelemetry. https://modelcontextprotocol.io/specification/2026-07-28/deprecated
- CONFIRMED: legacy HTTP+SSE transport deprecated; use Streamable HTTP. Transports: stdio (local) and Streamable HTTP (remote, OAuth recommended).
- CONFIRMED: architecture. Host (the AI app) creates one client per server, enforces consent; client is 1:1 with a server; server exposes Tools, Resources, Prompts. Servers must not see the whole conversation or other servers. https://modelcontextprotocol.io/docs/learn/architecture
- CONFIRMED: roadmap (Aug 22, 2026): agentic messaging primitives, HTTP-native transport unification, agent identity (DPoP, workload identity federation), progressive tool discovery. https://blog.modelcontextprotocol.io/posts/mcp-roadmap/
- UNCONFIRMED as official framing: "MCP vs function calling". Safe framing (ours, not the spec's): function calling is the model-to-application interface for choosing a tool; MCP is the application-to-tool-server interface for discovering and invoking tools/resources/prompts over a standard wire protocol. The host merges tools from all servers into one registry the model sees.

## A2A protocol

- CONFIRMED: created by Google (Apr 2025), donated to the Linux Foundation (June 23, 2025), and as of Aug 2026 hosted by the Agentic AI Foundation (AAIF, an LF foundation formed Dec 2025 that also hosts MCP, goose, AGENTS.md). https://a2a-protocol.org/latest/ ; https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation
- CONFIRMED: spec v1.0.0 (Mar 12, 2026); latest v1.0.1 (May 28, 2026). v1.0 flattened `Part`, added `tasks/list`, modernized OAuth (device code, PKCE). https://a2a-protocol.org/latest/specification/
- CONFIRMED core concepts: Agent Card (JSON metadata: identity, capabilities, skills, endpoint, auth; served at `agent-card.json`); Task (stateful unit of work; states SUBMITTED, WORKING, COMPLETED, FAILED, CANCELED, INPUT_REQUIRED, REJECTED, AUTH_REQUIRED); Message (role user|agent, Parts); Artifact (task output); streaming via SSE; push notifications via client webhook.
- CONFIRMED: "MCP is vertical" (one agent's capabilities), "A2A is horizontal" (agents partnering). https://a2a-protocol.org/latest/topics/a2a-and-mcp/ . Note: MCP's Tasks extension and roadmap narrow this gap.

## Anthropic engineering guidance

- "Writing effective tools for agents" (Sep 11, 2025) https://www.anthropic.com/engineering/writing-tools-for-agents : build evals from real multi-step workflows; track accuracy, runtime, tool-call count, tokens, errors; namespace tools by service/resource (`asana_search`); consolidate (`schedule_event` over three primitive calls; `search_logs` over `read_logs`); return semantic fields not UUIDs; offer concise vs detailed response formats; paginate/truncate (Claude Code caps tool results at 25,000 tokens); actionable error text; descriptions written like onboarding a new hire; unambiguous parameter names (`user_id` not `user`).
- "Effective context engineering for AI agents" (Sep 29, 2025) https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents : context is finite ("context rot"); system prompts at the right altitude; minimal non-overlapping tool sets; few canonical examples; just-in-time retrieval via lightweight identifiers; compaction preserving decisions and open issues; structured note-taking (NOTES.md, todo lists); sub-agents returning condensed 1,000-2,000 token summaries.
- "Building effective agents" (Dec 19, 2024) https://www.anthropic.com/engineering/building-effective-agents : workflows = predefined code paths; agents = LLM directs its own process; start with the augmented LLM; patterns: prompt chaining, routing, parallelization (sectioning/voting), orchestrator-workers, evaluator-optimizer; agents only for open-ended problems; simplicity, transparency, careful agent-computer interface (ACI).
- "Effective harnesses for long-running agents" (Nov 26, 2025) https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents : initializer agent + per-session coding agent; progress file and git as durable state; feature list with pass/fail; leave clean committed state; end-to-end verification.
- "Harness design for long-running application development" (Mar 24, 2026) https://www.anthropic.com/engineering/harness-design-long-running-apps : Planner / Generator / Evaluator; "sprint contracts" define done before coding; separate generation from evaluation (self-praise unreliable); gradable rubrics; fresh context resets with handoff artifacts beat compaction; remove harness components as models improve.
- "Scaling Managed Agents: decoupling the brain from the hands" (Apr 8, 2026) https://www.anthropic.com/engineering/managed-agents : session (append-only durable event log) / harness (stateless loop) / sandbox (interchangeable hands); crashes surface as tool errors; credentials never enter the sandbox.
- "Demystifying evals for AI agents" (Jan 9, 2026) on the same engineering index.

## Chain-of-thought faithfulness

- CONFIRMED: "Reasoning Models Don't Always Say What They Think" (Apr 3, 2025) https://www.anthropic.com/research/reasoning-models-dont-say-think : with injected hints, Claude 3.7 Sonnet verbalized the hint 25% of the time, DeepSeek R1 39%; outcome-based RL plateaued faithfulness around 20-28%; in a reward-hacking setup models exploited the hack >99% of the time and admitted it <2%. Lesson: CoT text is a useful signal, not an audit log; verify outcomes.

## OpenAI

- CONFIRMED reasoning guidance https://developers.openai.com/api/docs/guides/reasoning : give task, constraints, output format; treat reasoning effort as a tuning knob; for agentic work "define what counts as done and how the model should verify its work"; avoid chain-of-thought prefixes. GPT-5 prompting guide: control "agentic eagerness" (tool-call budgets, escape hatches), tool preambles, explicit stop conditions, remove contradictory instructions.
- CONFIRMED OpenAI Agents SDK (Python) v0.22.3 (Sep 17, 2026): Agents, Handoffs, Guardrails, Sessions (SQLite, Redis, SQLAlchemy, encrypted...), Tracing, MCP support, realtime/voice, Sandbox agents (persistent workspaces, Docker/local/hosted, resumable). https://openai.github.io/openai-agents-python/
- CONFIRMED (date secondary-sourced): OpenAI Agents API public beta (~Sep 10, 2026): hosted Codex harness via API; Agent, Environment (sandbox), durable Session, Events; OpenAI handles orchestration, context compaction, recovery. https://developers.openai.com/api/docs/guides/agents-api/overview
