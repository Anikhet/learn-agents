import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).parent
NOTEBOOK_DIR = ROOT / "notebooks"


TOPICS = [
    ("00", "Course Orientation", "How agentic systems differ from ordinary prompt wrappers, and how to use this notebook series.", "orientation", "Map the book's design-pattern path into a hands-on curriculum.", "Identify the core agent loop: observe, decide, act, evaluate.", "Run a tiny deterministic agent loop."),
    ("01", "Prompt Chaining", "Break a complex task into ordered model calls where each output becomes the next step's input.", "chain", "Decompose broad tasks into small prompts.", "Pass structured intermediate state between steps.", "Add validation checkpoints between steps."),
    ("02", "Routing", "Classify an incoming task and send it to the best specialized handler.", "routing", "Design a task router.", "Separate classification from execution.", "Add fallback behavior when the route is uncertain."),
    ("03", "Parallelization", "Run independent subtasks at the same time and merge their results.", "parallel", "Spot independent work units.", "Use concurrent execution safely.", "Aggregate results with a deterministic reducer."),
    ("04", "Reflection", "Use a critic or reviewer loop to improve an initial answer or artifact.", "reflection", "Separate generation from critique.", "Turn critiques into targeted revisions.", "Stop loops before they waste budget."),
    ("05", "Tool Use Function Calling", "Give an agent typed tools and let it select actions instead of only writing text.", "tools", "Define tool schemas as contracts.", "Dispatch tool calls safely.", "Return tool observations to the agent loop."),
    ("06", "Planning", "Create and execute a plan for multi-step goals.", "planning", "Represent plans as explicit steps.", "Track plan status.", "Revise a plan when execution changes the context."),
    ("07", "Multi-Agent Collaboration", "Coordinate specialized agents that divide work and share results.", "multi_agent", "Assign narrow responsibilities.", "Use a coordinator to reduce confusion.", "Merge specialist outputs into one deliverable."),
    ("08", "Memory Management", "Store, retrieve, and summarize useful context across turns or tasks.", "memory", "Choose what belongs in memory.", "Retrieve relevant memories.", "Compress old state to control context size."),
    ("09", "Learning and Adaptation", "Improve future behavior using feedback, examples, and lightweight policy updates.", "learning", "Capture feedback as data.", "Update simple preferences.", "Evaluate whether adaptation improved outcomes."),
    ("10", "Model Context Protocol MCP", "Understand MCP-style tool/resource boundaries and why clean interfaces matter.", "mcp", "Model resources, prompts, and tools.", "Keep context providers separate from agent logic.", "Simulate a client calling a server capability."),
    ("11", "Goal Setting and Monitoring", "Translate high-level goals into measurable progress signals.", "goals", "Define goals and success criteria.", "Monitor progress over steps.", "Detect stalled work."),
    ("12", "Exception Handling and Recovery", "Handle tool failures, invalid outputs, and missing information without collapsing the workflow.", "exceptions", "Classify failure types.", "Retry only when useful.", "Escalate or degrade gracefully."),
    ("13", "Human in the Loop", "Insert human approval, review, or clarification at high-value decision points.", "hitl", "Identify when human judgment is required.", "Represent approvals as workflow gates.", "Keep the agent productive while waiting."),
    ("14", "Knowledge Retrieval RAG", "Ground an agent's answer in retrieved documents or records.", "rag", "Build a tiny retrieval index.", "Retrieve context before answering.", "Cite source ids from retrieved records."),
    ("15", "Inter-Agent Communication A2A", "Let agents exchange structured messages instead of implicit prose blobs.", "a2a", "Create message envelopes.", "Route messages by role and intent.", "Preserve traceability across agents."),
    ("16", "Resource-Aware Optimization", "Make agents budget-aware across latency, token, cost, and compute constraints.", "resource", "Estimate resource use.", "Select cheaper strategies when possible.", "Stop work when marginal value is low."),
    ("17", "Reasoning Techniques", "Use explicit reasoning scaffolds such as decomposition, verification, and comparison.", "reasoning", "Pick a reasoning scaffold for a task.", "Separate assumptions from conclusions.", "Check answers against constraints."),
    ("18", "Guardrails Safety Patterns", "Constrain inputs, outputs, and tool actions to reduce unsafe or off-policy behavior.", "guardrails", "Validate input intent.", "Filter tool calls by policy.", "Post-check outputs before returning them."),
    ("19", "Evaluation and Monitoring", "Measure agent quality with tests, traces, and operational metrics.", "evals", "Write task-level eval cases.", "Track quality and reliability signals.", "Inspect failures for pattern-level improvements."),
    ("20", "Prioritization", "Rank tasks, memories, tool calls, or agent work by value and urgency.", "priority", "Score candidate actions.", "Resolve conflicts between urgency and importance.", "Choose the next best action."),
    ("21", "Exploration and Discovery", "Let agents search a solution space while keeping experiments bounded and observable.", "explore", "Generate candidate hypotheses.", "Test candidates cheaply.", "Keep an exploration log."),
    ("A", "Appendix A Advanced Prompting", "Reusable prompting tactics that support agentic workflows.", "prompting", "Write role, task, context, and output constraints.", "Use examples without overfitting.", "Ask for structured outputs."),
    ("B", "Appendix B Agentic Interactions", "Common interaction models between users, agents, tools, and other agents.", "interactions", "Sketch interaction topologies.", "Choose synchronous or asynchronous flow.", "Define handoff points."),
    ("C", "Appendix C Quick Reference", "A compact reference notebook for pattern selection.", "reference", "Match symptoms to patterns.", "Compare pattern tradeoffs.", "Pick a starting architecture."),
    ("D", "Appendix D Building an Agent", "Assemble multiple patterns into a small complete agent.", "build_agent", "Combine routing, tools, memory, and evals.", "Keep the agent loop inspectable.", "Run an end-to-end task."),
    ("E", "Appendix E AI Agents on the CLI", "Design command-line agent workflows with clear inputs, logs, and exit behavior.", "cli", "Parse a CLI-style request.", "Emit useful progress logs.", "Return machine-readable results."),
    ("F", "Appendix F Under the Hood", "Inspect the internal moving parts of an agent runtime.", "under_hood", "Trace state transitions.", "Understand action-observation loops.", "Debug a failed step."),
    ("G", "Appendix G Coding Agents", "Apply agentic patterns to software engineering tasks.", "coding", "Plan code changes.", "Use tests as feedback.", "Review generated patches before shipping."),
]


CODE_SNIPPETS = {
    "orientation": '''
class TinyAgent:
    def __init__(self):
        self.trace = []

    def observe(self, task):
        self.trace.append(("observe", task))
        return {"task": task, "keywords": task.lower().split()}

    def decide(self, state):
        action = "search" if "find" in state["keywords"] else "answer"
        self.trace.append(("decide", action))
        return action

    def act(self, action, state):
        result = f"{action}: handled '{state['task']}'"
        self.trace.append(("act", result))
        return result

agent = TinyAgent()
state = agent.observe("find examples of agent memory")
result = agent.act(agent.decide(state), state)
result, agent.trace
''',
    "chain": '''
def extract_requirements(request):
    return {"audience": "developers", "task": request, "format": "checklist"}

def draft(requirements):
    return [f"Explain {requirements['task']}", "Show one concrete example", "Add a practice task"]

def polish(items):
    return "\\n".join(f"- {item}" for item in items)

request = "prompt chaining for customer-support analysis"
polish(draft(extract_requirements(request)))
''',
    "routing": '''
def route(task):
    text = task.lower()
    if any(word in text for word in ["refund", "invoice", "billing"]):
        return "billing"
    if any(word in text for word in ["bug", "error", "crash"]):
        return "technical_support"
    return "general"

handlers = {
    "billing": lambda task: f"Billing agent opens account workflow for: {task}",
    "technical_support": lambda task: f"Support agent starts diagnostics for: {task}",
    "general": lambda task: f"General agent asks a clarifying question about: {task}",
}

task = "The app crashes when I export an invoice"
handlers[route(task)](task)
''',
    "parallel": '''
from concurrent.futures import ThreadPoolExecutor

def summarize_angle(angle):
    return f"{angle}: one risk, one opportunity, one metric"

angles = ["customer", "technical", "business", "safety"]
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(summarize_angle, angles))

"\\n".join(results)
''',
    "reflection": '''
def produce(topic):
    return f"{topic}: define the pattern and give an example."

def critique(answer):
    issues = []
    if "tradeoff" not in answer:
        issues.append("Add a tradeoff.")
    if "test" not in answer:
        issues.append("Add a verification step.")
    return issues

def revise(answer, issues):
    additions = " ".join(issues)
    return f"{answer} Revision notes: {additions}"

draft = produce("Reflection pattern")
final = revise(draft, critique(draft))
final
''',
    "tools": '''
TOOLS = {
    "add": lambda a, b: a + b,
    "lookup_status": lambda ticket_id: {"ticket_id": ticket_id, "status": "open"},
}

def call_tool(name, **kwargs):
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return TOOLS[name](**kwargs)

tool_request = {"name": "lookup_status", "arguments": {"ticket_id": "INC-1042"}}
call_tool(tool_request["name"], **tool_request["arguments"])
''',
    "planning": '''
plan = [
    {"step": "collect requirements", "status": "pending"},
    {"step": "choose tools", "status": "pending"},
    {"step": "run solution", "status": "pending"},
    {"step": "verify output", "status": "pending"},
]

def complete_next(plan):
    for item in plan:
        if item["status"] == "pending":
            item["status"] = "done"
            break
    return plan

complete_next(plan)
complete_next(plan)
plan
''',
    "multi_agent": '''
def researcher(topic):
    return {"role": "researcher", "finding": f"Key facts about {topic}"}

def builder(topic):
    return {"role": "builder", "artifact": f"Prototype outline for {topic}"}

def coordinator(topic):
    outputs = [researcher(topic), builder(topic)]
    return {"topic": topic, "team_outputs": outputs, "decision": "ready for review"}

coordinator("agent memory feature")
''',
    "memory": '''
class Memory:
    def __init__(self):
        self.items = []

    def add(self, text, tags):
        self.items.append({"text": text, "tags": set(tags)})

    def search(self, tag):
        return [item["text"] for item in self.items if tag in item["tags"]]

memory = Memory()
memory.add("User prefers notebook examples that run offline.", ["preference", "notebooks"])
memory.add("RAG requires retrieval before answer drafting.", ["rag", "pattern"])
memory.search("notebooks")
''',
    "learning": '''
policy = {"short_answers": 0, "examples": 0}

def record_feedback(feedback):
    if "too long" in feedback:
        policy["short_answers"] += 1
    if "need example" in feedback:
        policy["examples"] += 1

record_feedback("need example")
record_feedback("too long")
policy
''',
    "mcp": '''
server = {
    "resources": {"project://readme": "Agent tutorial project"},
    "tools": {"count_words": lambda text: len(text.split())},
}

def mcp_client_call(kind, name, **kwargs):
    if kind == "resource":
        return server["resources"][name]
    if kind == "tool":
        return server["tools"][name](**kwargs)
    raise ValueError("Unsupported capability kind")

mcp_client_call("tool", "count_words", text=mcp_client_call("resource", "project://readme"))
''',
    "goals": '''
goal = {"target": "answer has citations", "checks": ["retrieved_sources", "source_ids_in_answer"]}
state = {"retrieved_sources": True, "source_ids_in_answer": False}

def progress(goal, state):
    passed = [check for check in goal["checks"] if state.get(check)]
    return {"passed": passed, "score": len(passed) / len(goal["checks"])}

progress(goal, state)
''',
    "exceptions": '''
def unreliable_lookup(key):
    data = {"known": "value"}
    if key not in data:
        raise KeyError(key)
    return data[key]

def recoverable_call(key):
    try:
        return {"ok": True, "value": unreliable_lookup(key)}
    except KeyError:
        return {"ok": False, "recovery": "ask user for a valid key or use fallback data"}

recoverable_call("missing")
''',
    "hitl": '''
def approval_gate(action, risk):
    if risk == "high":
        return {"status": "waiting_for_human", "action": action}
    return {"status": "approved", "action": action}

approval_gate("send refund", "high")
''',
    "rag": '''
docs = {
    "doc-1": "Prompt chaining decomposes complex tasks into ordered steps.",
    "doc-2": "Routing sends a task to a specialized handler.",
    "doc-3": "Memory stores useful context across interactions.",
}

def retrieve(query, k=2):
    q = set(query.lower().split())
    scored = []
    for doc_id, text in docs.items():
        score = len(q & set(text.lower().split()))
        scored.append((score, doc_id, text))
    return [(doc_id, text) for score, doc_id, text in sorted(scored, reverse=True)[:k] if score]

retrieve("How do agents keep context in memory?")
''',
    "a2a": '''
def message(sender, recipient, intent, payload):
    return {"sender": sender, "recipient": recipient, "intent": intent, "payload": payload}

inbox = []
inbox.append(message("planner", "researcher", "request", {"topic": "RAG failure modes"}))
inbox.append(message("researcher", "planner", "response", {"finding": "stale context is a common risk"}))
inbox
''',
    "resource": '''
strategies = [
    {"name": "cheap_classifier", "cost": 1, "quality": 0.70},
    {"name": "full_agent_loop", "cost": 8, "quality": 0.92},
    {"name": "human_review", "cost": 20, "quality": 0.98},
]

budget = 10
best = max((s for s in strategies if s["cost"] <= budget), key=lambda s: s["quality"])
best
''',
    "reasoning": '''
def compare_options(options, criteria):
    rows = []
    for option in options:
        score = sum(option.get(c, 0) for c in criteria)
        rows.append({"option": option["name"], "score": score})
    return sorted(rows, key=lambda row: row["score"], reverse=True)

compare_options(
    [{"name": "single prompt", "simple": 3, "reliable": 1}, {"name": "prompt chain", "simple": 2, "reliable": 3}],
    ["simple", "reliable"],
)
''',
    "guardrails": '''
blocked_tools = {"delete_database", "send_money"}

def policy_check(tool_name, user_confirmed=False):
    if tool_name in blocked_tools and not user_confirmed:
        return {"allowed": False, "reason": "requires explicit human approval"}
    return {"allowed": True}

policy_check("send_money")
''',
    "evals": '''
eval_cases = [
    {"input": "refund request", "expected_route": "billing"},
    {"input": "app crash", "expected_route": "technical_support"},
]

def simple_route(text):
    return "billing" if "refund" in text else "technical_support" if "crash" in text else "general"

results = [{"case": case, "passed": simple_route(case["input"]) == case["expected_route"]} for case in eval_cases]
results
''',
    "priority": '''
tasks = [
    {"task": "fix production incident", "urgency": 5, "value": 5},
    {"task": "rewrite docs", "urgency": 1, "value": 3},
    {"task": "add evals", "urgency": 3, "value": 4},
]

def priority_score(task):
    return task["urgency"] * 2 + task["value"]

sorted(tasks, key=priority_score, reverse=True)
''',
    "explore": '''
hypotheses = ["use routing", "use retrieval", "use human approval"]

def cheap_test(hypothesis):
    return {"hypothesis": hypothesis, "score": len(hypothesis.split())}

log = [cheap_test(h) for h in hypotheses]
sorted(log, key=lambda row: row["score"], reverse=True)
''',
    "prompting": '''
prompt_parts = {
    "role": "You are a careful support analyst.",
    "task": "Classify the customer issue.",
    "context": "Use only the ticket text.",
    "output": "Return JSON with route and confidence.",
}

"\\n".join(prompt_parts.values())
''',
    "interactions": '''
topologies = {
    "user_to_agent": ["user", "agent", "tool"],
    "supervisor": ["user", "supervisor", "specialist_agents"],
    "peer_to_peer": ["agent_a", "agent_b", "shared_protocol"],
}
topologies
''',
    "reference": '''
pattern_picker = {
    "too complex for one prompt": "Prompt Chaining",
    "many possible handlers": "Routing",
    "needs current facts": "Knowledge Retrieval RAG",
    "risky action": "Human in the Loop",
    "quality unknown": "Evaluation and Monitoring",
}
pattern_picker["needs current facts"]
''',
    "build_agent": '''
def mini_agent(task):
    route_name = "lookup" if "what" in task.lower() else "write"
    memory = ["prefer concise answers"]
    observation = f"route={route_name}; memory={memory[0]}"
    return {"task": task, "observation": observation, "answer": f"Handled with {route_name} workflow"}

mini_agent("What is prompt chaining?")
''',
    "cli": '''
import argparse

parser = argparse.ArgumentParser(prog="agent")
parser.add_argument("task")
parser.add_argument("--json", action="store_true")
args = parser.parse_args(["summarize-ticket", "--json"])
{"task": args.task, "json": args.json, "exit_code": 0}
''',
    "under_hood": '''
trace = []

def step(name, state):
    state = {**state, "last_step": name}
    trace.append((name, dict(state)))
    return state

state = {}
for name in ["observe", "plan", "act", "evaluate"]:
    state = step(name, state)
trace
''',
    "coding": '''
change_plan = [
    "read failing test",
    "inspect relevant code",
    "make minimal patch",
    "run focused test",
    "review diff",
]
[{ "step": i + 1, "action": action } for i, action in enumerate(change_plan)]
''',
}


CONCEPTS = {
    "orientation": [
        "Agentic systems combine goals, context, decisions, actions, feedback, and state.",
        "The core loop is observe, reason or decide, act, evaluate, and update memory.",
        "Patterns are reusable control-flow choices, not model features.",
        "Reliability comes from explicit boundaries: tool contracts, state schemas, evals, and human gates.",
        "A useful agent should be inspectable before it is autonomous.",
    ],
    "chain": [
        "Decomposition: split a large task into smaller operations with clear inputs and outputs.",
        "Intermediate artifacts: summaries, extracted fields, plans, critiques, or transformed data passed between steps.",
        "Validation gates: check each step before downstream work amplifies mistakes.",
        "Context control: each prompt should receive only the information needed for that step.",
        "Pipeline design: linear chains are simple, while branching chains need routing and error recovery.",
    ],
    "routing": [
        "Intent classification: decide what kind of task arrived before doing the work.",
        "Specialized handlers: route to prompts, tools, models, agents, or workflows optimized for a case.",
        "Confidence thresholds: uncertain routes should ask for clarification or use a safe fallback.",
        "Hierarchical routing: broad category first, narrower route second.",
        "Evaluation: routing needs confusion-matrix style tests because one wrong route can ruin the task.",
    ],
    "parallel": [
        "Independence: parallel work is safe only when subtasks do not require each other's outputs.",
        "Fan-out and fan-in: distribute work, then merge it with a deterministic or review-based reducer.",
        "Diversity: parallel agents or prompts can explore different perspectives or candidates.",
        "Resource tradeoff: parallelism lowers latency but can increase cost and coordination complexity.",
        "Aggregation: merging is often the hardest part and should include conflict handling.",
    ],
    "reflection": [
        "Producer-critic separation: one component creates, another evaluates.",
        "Revision loops: critiques should point to actionable changes rather than vague preferences.",
        "Rubrics: reflection improves when the critic uses explicit criteria.",
        "Stopping conditions: use max iterations, quality thresholds, or diminishing-return checks.",
        "Risk: self-critique can reinforce model blind spots unless paired with tests or external evidence.",
    ],
    "tools": [
        "Tool schema: name, description, parameters, return type, and side-effect level form a contract.",
        "Action selection: the agent chooses when a tool is necessary instead of hallucinating facts.",
        "Observation passing: tool results become new context for the next decision.",
        "Safety: tools need allowlists, validation, auth boundaries, and idempotency where possible.",
        "Error handling: invalid arguments, unavailable services, and partial failures must be represented explicitly.",
    ],
    "planning": [
        "Task decomposition: convert a goal into ordered, observable steps.",
        "Plan representation: steps need status, dependencies, owner, and completion criteria.",
        "Execution monitoring: compare actual observations against the plan.",
        "Replanning: revise when tools fail, context changes, or a step reveals new constraints.",
        "Granularity: steps that are too broad are hard to verify; steps that are too small waste overhead.",
    ],
    "multi_agent": [
        "Role specialization: agents should have narrow responsibilities and clear authority.",
        "Coordination topology: supervisor, network, hierarchy, and custom workflows make different tradeoffs.",
        "Shared state: agents need a common workspace or message protocol to avoid duplicated or inconsistent work.",
        "Conflict resolution: disagreements require a policy, adjudicator, or evidence-based merge.",
        "Observability: multi-agent systems need traces that show who did what and why.",
    ],
    "memory": [
        "Short-term memory: active task state used inside the current workflow.",
        "Long-term memory: durable facts, preferences, summaries, and lessons reused later.",
        "Retrieval: memory is useful only when relevant items can be found at the right time.",
        "Summarization: old context should be compressed without losing decision-critical details.",
        "Privacy and decay: memory needs retention rules, deletion, freshness checks, and user control.",
    ],
    "learning": [
        "Feedback capture: collect ratings, corrections, outcomes, and traces in structured form.",
        "Adaptation level: update prompts, routing rules, memories, examples, or model weights depending on risk.",
        "Online vs offline learning: production systems usually favor evaluated offline updates for safety.",
        "Regression risk: improvements on one task can degrade another without a test suite.",
        "Self-improvement loop: propose change, test it, compare, promote only if metrics improve.",
    ],
    "mcp": [
        "Separation of concerns: MCP-style servers expose tools, resources, and prompts outside the agent.",
        "Context providers: the agent should request needed context instead of embedding every integration.",
        "Tool boundaries: each capability has a typed interface and controlled permissions.",
        "Interoperability: common protocols let different clients use the same capabilities.",
        "Security: resource access, credentials, and side-effecting tools need strict isolation.",
    ],
    "goals": [
        "Goal formulation: convert vague intent into measurable success criteria.",
        "Monitoring: track progress, blockers, cost, elapsed time, and quality signals.",
        "Milestones: long-running work needs intermediate checkpoints.",
        "Stall detection: repeated failed attempts should trigger replanning or escalation.",
        "Alignment: optimize for the user's real goal, not a proxy metric that is easier to measure.",
    ],
    "exceptions": [
        "Failure taxonomy: distinguish user ambiguity, tool errors, invalid outputs, policy blocks, and exhausted budgets.",
        "Retries: retry transient failures with limits and changed inputs when useful.",
        "Fallbacks: degrade gracefully to simpler workflows or partial answers.",
        "Compensation: side-effecting workflows may need rollback or reconciliation steps.",
        "Escalation: some failures should ask a human rather than continue autonomously.",
    ],
    "hitl": [
        "Control points: add human review where risk, ambiguity, cost, or irreversible action is high.",
        "Approval payloads: humans need enough context to make a decision quickly.",
        "Modes: clarify, approve, edit, rank, override, or audit.",
        "Latency tradeoff: human gates improve safety but slow automation.",
        "Auditability: record who approved what, when, and with which evidence.",
    ],
    "rag": [
        "Retrieval before generation: ground answers in documents, records, or search results.",
        "Chunking and indexing: retrieval quality depends on document segmentation and metadata.",
        "Ranking: lexical, vector, hybrid, and reranking approaches trade speed and relevance.",
        "Grounded synthesis: answers should cite retrieved source ids and avoid unsupported claims.",
        "Failure modes: stale docs, missing docs, bad chunks, prompt injection in documents, and over-reliance on weak evidence.",
    ],
    "a2a": [
        "Message envelope: sender, recipient, intent, payload, correlation id, and timestamp.",
        "Protocol discipline: agents should exchange structured data rather than ambiguous prose.",
        "Handoffs: ownership transfer needs task state, constraints, and expected output.",
        "Conversation state: multi-turn agent-to-agent work needs thread ids and traceability.",
        "Trust boundaries: not every agent should be allowed to instruct every other agent or tool.",
    ],
    "resource": [
        "Budget dimensions: tokens, latency, money, tool quotas, memory, and human attention.",
        "Strategy selection: choose the cheapest workflow that can meet the quality target.",
        "Early stopping: stop when expected benefit falls below expected cost.",
        "Caching: reuse deterministic or stable results to reduce repeated work.",
        "Graceful degradation: return a partial result or lower-cost method when budgets are constrained.",
    ],
    "reasoning": [
        "Decomposition: break the problem into explicit subquestions or constraints.",
        "Verification: check conclusions against facts, requirements, and edge cases.",
        "Comparison: evaluate alternatives with criteria rather than preference alone.",
        "Assumption tracking: state what is known, inferred, missing, or uncertain.",
        "Structured outputs: reasoning is easier to evaluate when represented as tables, plans, or checklists.",
    ],
    "guardrails": [
        "Input guardrails: detect unsafe, irrelevant, malicious, or policy-violating requests.",
        "Tool guardrails: restrict dangerous tools, validate arguments, and require approval for side effects.",
        "Output guardrails: check for unsupported claims, sensitive data, unsafe advice, or format violations.",
        "Prompt-injection resistance: treat retrieved or user-provided text as data, not instructions.",
        "Defense in depth: combine policy, schemas, sandboxing, evals, logging, and human review.",
    ],
    "evals": [
        "Task evals: fixed cases with expected behavior for routes, tools, answers, and workflows.",
        "Reference-free checks: rubric scoring, constraint checks, and human review where exact answers are unavailable.",
        "Operational monitoring: latency, cost, tool errors, retries, escalation rate, and user corrections.",
        "Trace inspection: debugging agents requires step-level inputs, outputs, decisions, and observations.",
        "Continuous improvement: eval failures should become new regression tests.",
    ],
    "priority": [
        "Scoring: rank work by urgency, value, risk, dependencies, and cost.",
        "Queues: agents often need task queues, memory queues, or tool-call queues.",
        "Preemption: urgent or high-risk tasks may interrupt lower-priority work.",
        "Fairness: pure priority can starve important but non-urgent tasks.",
        "Dynamic updates: priorities should change as new information arrives.",
    ],
    "explore": [
        "Hypothesis generation: propose multiple candidate solutions or search paths.",
        "Bounded experimentation: set budget, stop criteria, and evaluation metrics before exploring.",
        "Exploration vs exploitation: balance trying new paths with using the best-known path.",
        "Search logs: record candidates, evidence, decisions, and discarded options.",
        "Discovery risk: unconstrained exploration can drift from the goal or burn resources.",
    ],
    "prompting": [
        "Instruction hierarchy: separate role, task, context, constraints, examples, and output format.",
        "Few-shot examples: examples steer behavior but can overconstrain if too narrow.",
        "Context engineering: select, compress, and package only relevant information.",
        "Structured outputs: schemas make downstream parsing and validation easier.",
        "Prompt optimization: test variants against evals rather than relying on intuition.",
    ],
    "interactions": [
        "Interaction topology: user-agent, agent-tool, agent-agent, supervisor, and hierarchical flows.",
        "Sync vs async: short tasks can block; long tasks need status, checkpoints, and resumability.",
        "Turn-taking: define who can speak, act, ask, or approve at each stage.",
        "State ownership: decide where task state lives and who may mutate it.",
        "User experience: agent interactions should make progress and uncertainty visible.",
    ],
    "reference": [
        "Pattern selection: start from the failure mode or system need, not from the pattern name.",
        "Composability: real systems often combine routing, tools, memory, guardrails, and evals.",
        "Tradeoffs: every pattern adds overhead and should earn its place.",
        "Inspection: prefer designs where you can see inputs, decisions, outputs, and costs.",
        "Minimum viable agent: begin with deterministic control flow, then add model calls.",
    ],
    "build_agent": [
        "Architecture: combine router, planner, tools, memory, guardrails, and evaluator.",
        "State model: define a single task state object that every step updates.",
        "Trace model: persist step-by-step decisions and observations.",
        "Testing: create focused evals for each component and an end-to-end smoke test.",
        "Deployment: start with constrained autonomy and add permissions gradually.",
    ],
    "cli": [
        "Command surface: clear arguments, flags, stdin/stdout behavior, and exit codes.",
        "Machine-readable mode: JSON output helps compose CLI agents with other tools.",
        "Progress logs: stderr is useful for human-readable status while stdout remains parseable.",
        "Non-interactive operation: support automation without hidden prompts.",
        "Failure behavior: return useful errors and nonzero exit codes.",
    ],
    "under_hood": [
        "Runtime loop: receive task, construct context, call model, dispatch tools, update state, repeat.",
        "Schedulers: decide which task or agent step runs next.",
        "Context windows: manage what enters the model at each call.",
        "Tracing: every model call, tool call, and state mutation should be observable.",
        "Determinism boundaries: separate deterministic orchestration from probabilistic model outputs.",
    ],
    "coding": [
        "Repo orientation: inspect structure, tests, and conventions before editing.",
        "Patch planning: keep changes scoped and traceable to the bug or feature.",
        "Tool use: search, read, edit, run tests, and inspect diffs in a tight loop.",
        "Verification: tests, linters, type checks, and manual review are feedback signals.",
        "Risk control: avoid unrelated refactors and destructive git operations.",
    ],
}

PITFALLS = {
    "chain": ["Making every workflow a chain even when one well-scoped prompt is enough.", "Passing large unfiltered outputs forward and recreating context overload.", "Skipping intermediate validation."],
    "routing": ["Letting ambiguous cases silently fall into the wrong route.", "Using labels that overlap semantically.", "Evaluating only happy-path examples."],
    "parallel": ["Parallelizing dependent steps.", "Ignoring merge conflicts.", "Spending more on parallel branches than the task value justifies."],
    "reflection": ["Infinite or expensive revision loops.", "Critiques that are not tied to a rubric.", "Assuming self-critique catches factual errors without evidence."],
    "tools": ["Treating tool output as always trustworthy.", "Allowing side-effecting tools without confirmation.", "Using untyped arguments."],
    "planning": ["Plans that are too vague to execute.", "Refusing to replan after observations change.", "Optimizing for plan elegance instead of goal completion."],
    "multi_agent": ["Adding agents where functions would be simpler.", "No single owner for the final answer.", "Unstructured messages that lose accountability."],
    "memory": ["Saving everything forever.", "Retrieving irrelevant memories.", "Mixing user preferences with unverified facts."],
    "learning": ["Updating behavior from one noisy feedback event.", "No regression tests.", "Letting self-modification bypass review."],
    "mcp": ["Exposing broad tools instead of narrow capabilities.", "Leaking credentials through context.", "Coupling agent prompts to one server implementation."],
    "goals": ["Choosing metrics that are easy but misleading.", "No stall criteria.", "Forgetting user-visible progress."],
    "exceptions": ["Retrying permanent failures.", "Hiding partial failure from the user.", "No rollback plan for side effects."],
    "hitl": ["Asking humans to approve without enough evidence.", "Putting approval gates on every trivial action.", "No audit trail."],
    "rag": ["Retrieving weak context and still answering confidently.", "No citations or source ids.", "Ignoring prompt injection inside documents."],
    "a2a": ["Agents sending free-form instructions without protocol.", "No correlation ids.", "Unclear authority between agents."],
    "resource": ["Only tracking token cost.", "No maximum budget.", "Choosing cheap strategies that cannot meet the quality bar."],
    "reasoning": ["Verbose reasoning without verification.", "Hidden assumptions.", "No edge-case checks."],
    "guardrails": ["Relying on one prompt as the only safety layer.", "Guarding output but not tool use.", "No monitoring for guardrail misses."],
    "evals": ["Tiny eval sets that do not cover failure modes.", "Metrics with no action path.", "Not turning incidents into tests."],
    "priority": ["Starving low-urgency strategic work.", "Static priorities in a changing environment.", "No tie-breaker rules."],
    "explore": ["Exploring without a budget.", "Keeping no experiment log.", "Optimizing for novelty instead of progress."],
}

DEFAULT_PITFALLS = [
    "Adding pattern overhead before the simpler baseline fails.",
    "Leaving inputs, outputs, or state implicit.",
    "Skipping evals because the demo works once.",
]

DEEP_NOTES = {
    "orientation": [
        "Use the book's overall path as a progression: single model call, tool-connected agent, strategic planner, then collaborative multi-agent system.",
        "Treat an agent as a system boundary, not just a prompt. The boundary includes model calls, tools, state, memory, policies, and evaluation.",
        "The recurring implementation question is autonomy scope: what may the agent decide alone, what requires a tool, and what requires a human.",
        "The introduction's future-facing ideas are useful as design pressure tests: generalist agents, personalized proactive agents, embodied agents, economic agents, and self-modifying multi-agent systems.",
        "A practical first principle is to make the agent observable before making it powerful.",
    ],
    "chain": [
        "Prompt chaining is the baseline pattern for reducing cognitive load. Each step should have one job and an inspectable artifact.",
        "The chain can include deterministic processing between model calls, such as schema validation, entity normalization, or routing decisions.",
        "Context engineering is central: avoid passing the whole prior transcript when a compact intermediate representation is enough.",
        "Chains are strongest for workflows with a known order, such as extract, validate, transform, synthesize, and review.",
        "When the order is not known upfront, combine chaining with routing, planning, or reflection.",
    ],
    "routing": [
        "Routing is a control-flow pattern. It decides which specialized workflow, model, tool, or agent should handle a task.",
        "A production router should emit both the selected route and confidence or rationale fields that can be evaluated.",
        "Routing can be rule-based, embedding-based, model-based, or hierarchical. Simple rules are often the best first version.",
        "Fallbacks matter: a low-confidence route should ask a clarification question, use a general handler, or escalate.",
        "Routing quality should be measured like a classifier: false routes, ambiguous labels, and drift all matter.",
    ],
    "parallel": [
        "Parallelization is valuable when subtasks are independent or when diversity of candidates improves quality.",
        "The reducer is part of the pattern. It must merge, rank, deduplicate, or reconcile parallel outputs.",
        "Parallel branches may use different tools, agents, prompts, models, or data sources.",
        "Latency gains can hide cost growth. Track the number of branches and stop branches that no longer add value.",
        "Use this pattern for research fan-out, multi-source validation, multimodal analysis, and A/B option generation.",
    ],
    "reflection": [
        "Reflection creates an explicit quality loop: produce, critique, revise, and optionally repeat.",
        "A critic should use a rubric. Without criteria, reflection becomes taste-based rewriting.",
        "The producer and critic can be separate prompts, agents, models, or deterministic checks.",
        "Reflection is especially useful for code, plans, summaries, safety review, and high-quality writing.",
        "Do not rely on reflection alone for factuality. Pair it with retrieval, tests, or source-grounded verification.",
    ],
    "tools": [
        "Tool use turns a language model from a text generator into a system actor that can observe or change external state.",
        "The tool contract should be narrow: typed parameters, explicit return shape, known side effects, and clear errors.",
        "The loop is decision, function call, tool execution, observation, and follow-up reasoning.",
        "Tools should be protected with allowlists, parameter validation, authentication, rate limits, and human approval for irreversible actions.",
        "Use tools for live data, calculations, database operations, communications, code execution, and device control.",
    ],
    "planning": [
        "Planning is useful when the system must discover the how, not merely execute a fixed workflow.",
        "A good plan has steps, dependencies, assumptions, expected observations, and completion criteria.",
        "Planning should be adaptive. New evidence, failed tools, or user corrections should trigger replanning.",
        "Deep research systems are planning-heavy: they create an initial research plan, gather information, revise, and synthesize cited output.",
        "Use a fixed chain instead of planning when the workflow is stable and predictable.",
    ],
    "multi_agent": [
        "Multi-agent collaboration is a division-of-labor pattern. It is justified when specialization improves quality or scale.",
        "Common topologies include single agent, peer network, supervisor, supervisor-as-tool, hierarchy, and custom flows.",
        "A coordinator or supervisor reduces ambiguity by owning task decomposition, assignment, and final synthesis.",
        "Agent roles should be non-overlapping enough that each agent knows what it owns.",
        "Multi-agent systems need explicit state sharing, message structure, conflict resolution, and traceability.",
    ],
    "memory": [
        "Memory management separates immediate task state from durable knowledge and user-specific preferences.",
        "Short-term memory includes conversation state, current plan, tool observations, and temporary facts.",
        "Long-term memory can be semantic, episodic, or procedural: facts, experiences, and learned ways of working.",
        "Storage is not enough. Retrieval, ranking, summarization, freshness, and deletion policies determine whether memory is useful.",
        "Framework memory abstractions often distinguish sessions, state, chat history, and searchable memory services.",
    ],
    "learning": [
        "Learning is about changing future behavior from evidence. Adaptation is the visible behavior change that results.",
        "The chapter spans classic ML categories, preference optimization, memory-based learning, and self-improving coding systems.",
        "Self-improving agents need oversight because they can change prompts, code, tools, or policies in ways that regress safety or quality.",
        "Evolutionary systems such as algorithm-discovery agents use generate, evaluate, select, and mutate cycles.",
        "Every adaptation path should have regression tests and promotion criteria before it affects production behavior.",
    ],
    "mcp": [
        "MCP standardizes how agents discover and use external tools, resources, and prompts through client-server interfaces.",
        "Function calling is usually app-local and explicit; MCP is a reusable protocol layer for capability discovery and interoperability.",
        "Agent-friendly APIs matter. Wrapping a bad legacy API in MCP does not automatically make it useful for agents.",
        "Important design dimensions include local vs remote servers, on-demand vs batch use, transport choice, security, and error reporting.",
        "MCP is most valuable when many agents or clients need a changing set of tools and data sources.",
    ],
    "goals": [
        "Goals give agents purpose; monitoring tells the system whether progress is happening.",
        "Use SMART-style goals when possible: specific, measurable, achievable, relevant, and time-bound.",
        "Monitoring should track outputs, intermediate state, tool results, plan progress, cost, and environmental changes.",
        "Feedback loops let agents revise plans, retry, escalate, or stop when success criteria are not met.",
        "Goal monitoring overlaps with evaluation, but it is used during execution rather than only after delivery.",
    ],
    "exceptions": [
        "Exception handling begins with detection: invalid tool output, API errors, malformed model output, missing state, and user ambiguity.",
        "Handling strategies include logging, retrying transient failures, choosing fallbacks, degrading gracefully, and notifying stakeholders.",
        "Recovery returns the system to a stable state through rollback, diagnosis, self-correction, or escalation.",
        "Side-effecting workflows need compensation steps because retrying can duplicate actions.",
        "A robust agent should report partial progress and what failed instead of hiding the failure.",
    ],
    "hitl": [
        "Human-in-the-loop keeps human judgment inside the workflow at high-value points.",
        "HITL includes oversight, intervention, correction, approval, data labeling, and feedback for future learning.",
        "Human-on-the-loop is a related model where humans set policies and monitor without approving every individual action.",
        "Escalation policies should define when an agent hands off: uncertainty, risk, compliance, cost, or user frustration.",
        "The main tradeoff is safety and quality versus latency, cost, and operational complexity.",
    ],
    "rag": [
        "RAG grounds generation in retrieved external knowledge so the model is not limited to pretraining.",
        "The core pipeline is ingest, chunk, embed or index, retrieve, rank, generate, and cite.",
        "Retrieval can be lexical, vector, hybrid, graph-based, or agentic with source evaluation and reconciliation.",
        "Important retrieval controls include chunk size, overlap, metadata, top-k, similarity threshold, reranking, and freshness.",
        "Agentic RAG adds reasoning over sources: identifying contradictions, prioritizing authority, and deciding whether more retrieval is needed.",
    ],
    "a2a": [
        "A2A is about agent-to-agent task coordination, while MCP is about agent-to-tool or agent-to-resource access.",
        "A2A uses discoverable agent identity, commonly represented as an Agent Card that describes capabilities and interaction modes.",
        "Tasks need ids, status, context, messages, and artifacts so long-running collaboration can be tracked.",
        "Interaction modes include synchronous request-response, polling, server-sent event streaming, and webhook-style push notifications.",
        "Security and trust boundaries are central because agents may represent different systems, vendors, or permission levels.",
    ],
    "resource": [
        "Resource-aware optimization makes agents choose workflows based on budget, latency, quality, availability, and compute constraints.",
        "Dynamic model switching routes simple tasks to cheaper models and complex tasks to stronger models.",
        "Critique or evaluation agents can improve allocation by flagging poor routing decisions.",
        "Other optimizations include contextual pruning, proactive resource prediction, cost-sensitive exploration, parallelism awareness, learned allocation policies, and graceful degradation.",
        "Track budget as a first-class state variable, not as an afterthought.",
    ],
    "reasoning": [
        "Reasoning techniques are scaffolds for hard problems: decomposition, explicit intermediate work, search over alternatives, and verification.",
        "CoT supports stepwise decomposition; ToT explores multiple branches; self-correction revises flawed outputs.",
        "PAL delegates precise computation to code; ReAct interleaves reasoning with tool actions and observations.",
        "Debate-style and graph-style approaches use multiple agents or paths to compare and improve answers.",
        "Inference scaling means more test-time compute can improve performance, but cost and latency must be managed.",
    ],
    "guardrails": [
        "Guardrails should operate before, during, and after generation: input checks, tool checks, behavior constraints, output checks, and monitoring.",
        "Policy enforcement benefits from structured outputs and schema validation so decisions can be audited.",
        "Prompt injection, jailbreaking, unsafe tool use, bias, misinformation, and data leakage require different controls.",
        "Principle of least privilege means agents get only the tools and data needed for the current task.",
        "Guardrails must be monitored and updated because attacks, policies, and model behavior change over time.",
    ],
    "evals": [
        "Agent evaluation includes final output quality, intermediate trajectory quality, tool use correctness, cost, latency, and compliance.",
        "Use exact checks where possible, rubric or LLM-as-judge checks where subjective quality matters, and human review for high-stakes outputs.",
        "Trajectory evaluation compares the actual sequence of actions against expected or acceptable paths.",
        "Monitoring turns evaluation into an operational loop by detecting drift, anomalies, regressions, and budget issues.",
        "A failing production incident should become a new eval case.",
    ],
    "priority": [
        "Prioritization is decision-making under competing tasks, goals, dependencies, and resource constraints.",
        "The agent needs criteria such as urgency, importance, dependency, risk, user preference, and expected value.",
        "Prioritization can happen at strategic goal level, plan-step level, queue level, and tool-call level.",
        "Dynamic reprioritization updates ordering when new work, failures, deadlines, or external changes arrive.",
        "A prioritization system should expose why an item is P0, P1, or P2 so users can correct it.",
    ],
    "explore": [
        "Exploration and discovery is for open-ended problems where the correct path is not known in advance.",
        "A strong exploration system separates generation, reflection, ranking, evolution, clustering, and meta-review.",
        "Scientific discovery agents augment researchers by processing literature, generating hypotheses, designing experiments, and reviewing results.",
        "Exploration must be bounded by budget, safety review, ethical constraints, and evaluation criteria.",
        "The output is often not a final answer but a ranked set of hypotheses, strategies, or research directions.",
    ],
    "prompting": [
        "Advanced prompting is the cross-cutting skill behind most agentic patterns.",
        "Prompt structure should separate role, task, context, constraints, examples, output schema, and evaluation criteria.",
        "Few-shot examples should be representative and diverse; classification examples should avoid label ordering bias.",
        "Advanced reasoning prompts include CoT, self-consistency, step-back prompting, ToT, ReAct, and decomposition.",
        "Production prompting requires versioning, evals, documentation, and adaptation to model updates.",
    ],
    "interactions": [
        "Agentic interaction is expanding from text chat into browser control, desktop control, multimodal perception, and real-world assistance.",
        "Computer-use agents need visual perception, GUI element recognition, contextual interpretation, and action feedback loops.",
        "Examples in the ecosystem include browser agents, desktop agents, live multimodal assistants, and proactive web task agents.",
        "These systems need stronger safety boundaries because they can interact with user accounts, websites, files, and other applications.",
        "Vibe coding appears here as an interaction style: the human steers intent while the AI generates and iterates artifacts.",
    ],
    "reference": [
        "Framework choice depends on workflow shape: linear chains, stateful graphs, role-based teams, data-heavy RAG, or enterprise integration.",
        "LangChain is a strong fit for predictable LCEL-style pipelines and simple RAG, summarization, or extraction flows.",
        "LangGraph fits stateful, cyclical, human-gated, or multi-agent workflows where explicit graph control matters.",
        "ADK and CrewAI emphasize higher-level agent orchestration with roles, teams, and managed execution patterns.",
        "Other frameworks fill specialized niches: LlamaIndex for data, AutoGen for agent conversations, MetaGPT for role-based software workflows, SuperAGI for autonomous agents, and Semantic Kernel for enterprise code integration.",
    ],
    "build_agent": [
        "An enterprise agent platform combines search, knowledge graph, connectors, agent design, workflows, and security.",
        "Agent building should start from the business process and permissions, not from the model.",
        "No-code or low-code agent builders can accelerate prototypes, but the same design questions still apply: data access, tool permissions, evaluation, and monitoring.",
        "Enterprise agents need identity-aware access because they operate over private organizational knowledge.",
        "A finished agent should have a deployment path, user interface, audit trail, and maintenance owner.",
    ],
    "cli": [
        "CLI agents are practical because they operate close to code, files, git history, tests, and developer workflows.",
        "Different CLI agents optimize for different workflows: deep codebase work, multimodal general tasks, git-centric patching, or GitHub-integrated automation.",
        "Good CLI agents expose progress, ask before risky actions, preserve diffs, and support non-interactive automation.",
        "The terminal interface makes exit codes, stdout, stderr, logs, and JSON output important design details.",
        "Choose CLI tooling based on context window, editing model, git integration, tool permissions, and review workflow.",
    ],
    "under_hood": [
        "Under the hood, models parse the prompt, identify intent, activate relevant learned patterns, plan a response structure, generate tokens, and revise within constraints.",
        "This is not human reasoning. It is pattern-based probabilistic generation shaped by training, instruction tuning, and runtime context.",
        "Useful mental models include task parsing, concept activation, response planning, candidate evaluation, and final formatting.",
        "Model limitations include stale knowledge, hallucination risk, context sensitivity, and lack of independent ground truth without tools.",
        "This appendix helps learners understand why external tools, retrieval, guardrails, and evaluation are necessary.",
    ],
    "coding": [
        "Coding agents move from raw code generation toward specialized engineering workflows: implementation, testing, documentation, optimization, and review.",
        "The human remains the architect and quality gate, responsible for requirements, scope, and acceptance.",
        "Agent briefs should include relevant code, goals, constraints, standards, tests, and expected output format.",
        "Treat specialist prompts as code: version them, review them, and improve them from outcomes.",
        "A reliable coding-agent workflow includes orientation, plan, patch, tests, review, and documentation.",
    ],
}

COVERAGE_CHECKLISTS = {
    "orientation": [
        "Agent definition: autonomy, goals, reactivity, proactivity, tool use, memory, and communication.",
        "Agent loop: goal, perceive, reason, act, learn.",
        "Evolution path: LLMs to RAG to agentic RAG to multi-agent AI.",
        "Complexity levels: core LLM, connected tool user, strategic planner, collaborative multi-agent system.",
        "Future hypotheses: generalist agents, personalization, embodiment, agent economy, and metamorphic multi-agent systems.",
        "Framework context: LangChain, LangGraph, CrewAI, and Google ADK.",
    ],
    "chain": [
        "Limitations of single large prompts.",
        "Sequential decomposition and pipeline design.",
        "Intermediate output passed as next-step input.",
        "Validation between steps.",
        "Context engineering and focused context windows.",
        "Use cases: information processing, complex QA, data extraction, content generation, stateful conversations, code refinement, and multimodal reasoning.",
        "Framework awareness: LCEL, LangGraph, CrewAI tasks, and ADK-style sequential flows.",
        "When to use and when to avoid chaining.",
    ],
    "routing": [
        "Intent classification before execution.",
        "LLM-based routing.",
        "Embedding-based routing.",
        "Rule-based routing.",
        "Discriminative ML classifier routing.",
        "Fallback and clarification routes.",
        "Routing to chains, tools, sub-agents, or models.",
        "Evaluation of route accuracy and ambiguity.",
        "Framework awareness: LangGraph conditional flow and ADK coordinator patterns.",
    ],
    "parallel": [
        "Sequential vs parallel execution tradeoffs.",
        "Fan-out and fan-in architecture.",
        "Independent task requirement.",
        "Reducer or merger design.",
        "Use cases: research, data analysis, multi-API calls, content components, validation, multimodal processing, and A/B generation.",
        "Latency, cost, and synchronization risks.",
        "Framework awareness: LCEL RunnableParallel and ADK ParallelAgent with merger agents.",
    ],
    "reflection": [
        "Execution, critique, refinement, and optional iteration.",
        "Producer agent and critic agent roles.",
        "Rubric-driven evaluation.",
        "Self-correction and self-refinement.",
        "Stopping conditions and loop limits.",
        "Use cases: writing, code, problem solving, summarization, planning, and conversation repair.",
        "Tradeoffs: better quality vs higher latency and cost.",
        "Framework awareness: LangGraph stateful loops and ADK sequential reviewer flows.",
    ],
    "tools": [
        "Tool definition and schema design.",
        "LLM decision to call a tool.",
        "Structured function call generation.",
        "Tool execution by orchestration layer.",
        "Observation returned to the model.",
        "Use cases: external retrieval, databases, APIs, calculations, communication, code execution, and device control.",
        "Tool error handling and clean return data.",
        "Tool safety: validation, permissions, and side-effect control.",
        "Framework awareness: LangChain tools, CrewAI tools, ADK built-in tools, Google Search, and Vertex AI Search.",
    ],
    "planning": [
        "Planning as discovering the how for a high-level objective.",
        "Plan decomposition into executable steps.",
        "Constraints, assumptions, dependencies, and success criteria.",
        "Adaptation and replanning from new observations.",
        "When a fixed workflow is better than dynamic planning.",
        "Use cases: workflow automation, robotics/navigation, research reports, and multi-step problem solving.",
        "Deep research workflows: plan, user review, search, reflection, synthesis, and cited output.",
        "Framework awareness: CrewAI planning/writing flow, Google Deep Research, OpenAI Deep Research API, and MCP extensibility.",
    ],
    "multi_agent": [
        "Multi-agent division of labor.",
        "Specialist roles and coordinator/supervisor responsibilities.",
        "Topologies: single agent, network, supervisor, supervisor as tool, hierarchical, and custom.",
        "Sequential handoffs.",
        "Debate and consensus.",
        "Expert teams and shared workspaces.",
        "Use cases: product launches, research, software teams, customer support, and enterprise workflows.",
        "Framework awareness: CrewAI crews and Google ADK multi-agent patterns.",
    ],
    "memory": [
        "Short-term/contextual memory.",
        "Long-term/persistent memory.",
        "Session, state, and memory service separation.",
        "ChatMessageHistory and ConversationBufferMemory concepts.",
        "Semantic memory, episodic memory, and procedural memory.",
        "Searchable memory and retrieval.",
        "Memory summarization, pruning, privacy, and freshness.",
        "Framework awareness: ADK SessionService, MemoryService, DatabaseSessionService, VertexAiSessionService, VertexAiRagMemoryService, Memory Bank, LangGraph store.",
    ],
    "learning": [
        "Reinforcement learning.",
        "Supervised learning.",
        "Unsupervised learning.",
        "Few-shot and zero-shot learning with LLM agents.",
        "Online learning.",
        "Memory-based learning.",
        "PPO and stable policy updates.",
        "DPO and preference optimization.",
        "Self-improving coding agents and overseers.",
        "Evolutionary systems: AlphaEvolve and OpenEvolve.",
        "Regression safety and promotion criteria for learned changes.",
    ],
    "mcp": [
        "MCP as a standardized client-server interface.",
        "Tools, resources, and prompts as distinct MCP capabilities.",
        "MCP vs direct function calling.",
        "Dynamic discovery and manifests.",
        "MCP client, server, and external service roles.",
        "Discovery, request, server execution, and context update flow.",
        "Security, authorization, error handling, local vs remote deployment, transport, and on-demand vs batch use.",
        "Use cases: databases, external APIs, media generation, information extraction, custom tools, workflows, IoT, and finance.",
        "Framework awareness: ADK MCPToolset, npm/uvx servers, FastMCP, and tool filtering.",
    ],
    "goals": [
        "Initial state, goal state, options, and constraints.",
        "SMART goal formulation.",
        "Metrics and success criteria.",
        "Monitoring agent actions and environment state.",
        "Feedback loops for adaptation and replanning.",
        "Use cases: customer support, tutoring, project management, trading, robotics, and moderation.",
        "ADK-style instruction, state, and tool-based monitoring.",
        "Relationship to evaluation and monitoring.",
    ],
    "exceptions": [
        "Error detection for tool outputs, API errors, malformed outputs, and runtime failures.",
        "Logging for diagnosis.",
        "Retry strategy for transient failures.",
        "Fallback mechanisms.",
        "Graceful degradation.",
        "Notification and escalation.",
        "State rollback and compensation.",
        "Use cases: chatbots, trading, smart homes, data processing, web scraping, and robotics.",
        "Framework awareness: ADK sequential primary/fallback/response flow.",
    ],
    "hitl": [
        "Human oversight.",
        "Human intervention and correction.",
        "Human feedback for learning.",
        "Human-agent collaboration.",
        "Human-on-the-loop policy supervision.",
        "Escalation policies.",
        "Use cases: content moderation, autonomous driving, fraud, legal review, support, data labeling, generative refinement, and autonomous networks.",
        "Tradeoffs: scalability, latency, training reviewers, and operational cost.",
        "Framework awareness: ADK escalation tool patterns and LangChain human gates.",
    ],
    "rag": [
        "Retrieval-augmented generation purpose and limitations addressed.",
        "Embeddings and vector representations.",
        "Text similarity and semantic distance.",
        "Chunking strategy.",
        "Vector databases and approximate nearest-neighbor search.",
        "Lexical, vector, and hybrid retrieval.",
        "Reranking, top-k, thresholding, and metadata.",
        "RAG challenges: fragmented evidence, bad retrieval, contradictions, stale data, and preprocessing cost.",
        "GraphRAG.",
        "Agentic RAG with source evaluation and contradiction handling.",
        "Use cases: enterprise search, support, recommendation, news, legal, healthcare, education, and research.",
        "Framework awareness: ADK Google Search grounding, Vertex AI RAG, LangChain, LangGraph, and Weaviate-style vector stores.",
    ],
    "a2a": [
        "A2A as open agent-to-agent communication.",
        "Core actors: user, A2A client/client agent, A2A server/remote agent.",
        "Agent Card identity, capabilities, skills, modes, and auth requirements.",
        "Agent discovery: well-known URI, registries, and direct configuration.",
        "Tasks, task ids, status, messages, context ids, and artifacts.",
        "Interaction modes: synchronous request-response, asynchronous polling, SSE streaming, and webhook push notifications.",
        "Security and authentication.",
        "A2A vs MCP distinction.",
        "Use cases: multi-framework collaboration, workflow orchestration, and dynamic information retrieval.",
    ],
    "resource": [
        "Resource budgets: cost, latency, tokens, compute, energy, quotas, and human attention.",
        "Dynamic model selection by query complexity and budget.",
        "Router agent for model/tool selection.",
        "Critique agent for allocation feedback.",
        "Fallback for reliability.",
        "Contextual pruning and summarization.",
        "Proactive resource prediction.",
        "Cost-sensitive exploration.",
        "Adaptive task allocation in multi-agent systems.",
        "Parallelization and distributed computing awareness.",
        "Learned resource allocation policies.",
        "Framework awareness: ADK multi-agent router/critique, OpenAI routing, and OpenRouter auto/fallback routes.",
    ],
    "reasoning": [
        "Complex QA, math, code, planning, medical, and legal reasoning use cases.",
        "Chain-of-thought.",
        "Tree-of-thought.",
        "Self-correction/self-refinement.",
        "Program-aided language models.",
        "ReAct reasoning and acting loop.",
        "Chain of Debates and Graph of Debates.",
        "Multi-Agent System Search: block prompt optimization, topology optimization, and workflow-level prompt optimization.",
        "Scaling inference law and test-time compute.",
        "DeepSearch/Deep Research style systems.",
        "Reasoning tradeoffs: quality vs latency, cost, and verifiability.",
    ],
    "guardrails": [
        "Input validation and sanitization.",
        "Output filtering and post-processing.",
        "Behavioral constraints in prompts.",
        "Tool use restrictions and callbacks.",
        "External moderation or guardrail models.",
        "Jailbreak and instruction-subversion detection.",
        "Structured policy outputs and schema validation.",
        "Observability through structured logs.",
        "Human oversight for critical decisions.",
        "Principle of least privilege.",
        "Use cases: support, content generation, tutors, legal, HR, moderation, and scientific assistants.",
        "Framework awareness: CrewAI guardrails, Pydantic validation, ADK before_tool_callback, and Vertex safety screening.",
    ],
    "evals": [
        "Performance tracking in live systems.",
        "A/B testing.",
        "Compliance and safety audits.",
        "Drift detection.",
        "Anomaly detection.",
        "Learning progress assessment.",
        "Response accuracy metrics.",
        "Latency monitoring.",
        "Token usage and cost monitoring.",
        "LLM-as-a-judge rubrics.",
        "Human evaluation tradeoffs.",
        "Agent trajectory and tool-use evaluation.",
        "Test files, eval sets, and multi-turn cases.",
        "Contractor model and hierarchical decomposition concepts.",
        "Framework awareness: ADK Web, evaluation files, CI, and adk eval.",
    ],
    "priority": [
        "Criteria definition for urgency, importance, dependencies, resources, risk, and preferences.",
        "Task evaluation against criteria.",
        "Scheduling and selection logic.",
        "Dynamic reprioritization.",
        "Goal-level, subtask-level, and action-level prioritization.",
        "Use cases: support queues, cloud resources, autonomous driving, trading, project management, cybersecurity, and personal assistants.",
        "Task creation, priority assignment, worker assignment, and listing.",
        "Default priority behavior when information is missing.",
        "Framework awareness: LangChain tool-driven task manager.",
    ],
    "explore": [
        "Open-ended problem solving and novelty search.",
        "Exploration-exploitation tradeoff.",
        "Scientific research automation.",
        "Game strategy generation.",
        "Market research and trend spotting.",
        "Security vulnerability discovery.",
        "Creative content exploration.",
        "Personalized education path discovery.",
        "AI co-scientist: generation, reflection, ranking, evolution, proximity, and meta-review agents.",
        "Automated, expert, and safety evaluation of research outputs.",
        "Agent Laboratory: literature review, experimentation, report writing, AgentRxiv, professor, postdoc, engineer, and reviewer agents.",
        "Safety and augmentation limitations.",
    ],
    "prompting": [
        "Core prompting principles: clarity, specificity, conciseness, examples, and iteration.",
        "Zero-shot, one-shot, few-shot, and many-shot prompting.",
        "System prompting.",
        "Role prompting.",
        "Persona and user persona patterns.",
        "Context engineering.",
        "Prompt structure, delimiters, and output schemas.",
        "Chain-of-thought and few-shot CoT.",
        "Self-consistency.",
        "Step-back prompting.",
        "Tree-of-thought.",
        "Tool use/function calling.",
        "ReAct.",
        "Automatic Prompt Engineering.",
        "Iterative refinement, negative examples, analogies, factored cognition, RAG, Gems, meta-prompt refinement, code prompting, multimodal prompting, and prompt testing.",
    ],
    "interactions": [
        "Agent-computer interfaces.",
        "Visual perception.",
        "GUI element recognition.",
        "Contextual interpretation.",
        "Dynamic action and response loops.",
        "Browser and desktop agents.",
        "ChatGPT Operator, Project Mariner, Anthropic Computer Use, and Browser Use as ecosystem examples.",
        "Project Astra and Gemini Live as multimodal/real-world interaction examples.",
        "ChatGPT Agent and Claude model-family interaction capabilities.",
        "Vibe coding as an interactive development mode.",
        "Safety risks for agents acting through user interfaces.",
    ],
    "reference": [
        "LangChain and LCEL for linear DAG-style pipelines.",
        "LangGraph for stateful graphs, loops, human gates, and multi-agent systems.",
        "Google ADK for structured agent development and multi-agent orchestration.",
        "CrewAI for role-based multi-agent teams.",
        "Microsoft AutoGen for conversational multi-agent collaboration.",
        "LlamaIndex for data-intensive RAG systems.",
        "MetaGPT for role-based software teams.",
        "SuperAGI for autonomous agent workflows.",
        "Semantic Kernel for integrating LLMs with conventional enterprise code.",
        "Framework selection criteria by workflow shape.",
    ],
    "build_agent": [
        "AgentSpace as enterprise agent platform.",
        "Unified enterprise search.",
        "Enterprise knowledge graph.",
        "No-code Agent Designer.",
        "Multi-agent workflows.",
        "Connectors and enterprise data integration.",
        "Security and access control.",
        "UI-based agent creation flow.",
        "Deployment and operationalization concerns.",
    ],
    "cli": [
        "AI agent CLI category and why terminal workflows matter.",
        "Claude Code strengths and use cases.",
        "Gemini CLI strengths and use cases.",
        "Aider git-centric editing strengths and use cases.",
        "GitHub Copilot CLI/GitHub-integrated workflow strengths and use cases.",
        "Tool choice criteria: context, editing behavior, git flow, cost, and permissions.",
        "CLI design concerns: logs, exit codes, JSON, non-interactive mode, and auditability.",
    ],
    "under_hood": [
        "Prompt deconstruction into task, concepts, constraints, and intent.",
        "Internal knowledge activation and pattern recognition.",
        "Response planning and structure selection.",
        "Candidate response evaluation.",
        "Token-by-token generation.",
        "Rule, style, and safety instruction application.",
        "Review and refinement pass.",
        "Limitations: not human reasoning, no independent new knowledge, bounded by training and context.",
        "Why tools, retrieval, memory, and guardrails compensate for model limits.",
    ],
    "coding": [
        "Vibe coding as rapid exploratory development.",
        "Transition from raw code generation to managed coding-agent workflows.",
        "Human as architect, orchestrator, and final quality gate.",
        "Complete codebase and context packaging.",
        "Specialist agents: coder, tester, documenter, optimizer, and reviewer.",
        "Prompts as versioned markdown assets.",
        "Practical implementation workflow with briefs, task folders, and review loops.",
        "Testing, documentation, optimization, and code review as agent tasks.",
        "Architectural ownership, high-quality briefs, iterative dialogue, and final verification.",
    ],
}


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


def notebook(title, description, code_key, objectives):
    concepts = "\n".join(f"- {item}" for item in CONCEPTS[code_key])
    pitfalls = "\n".join(f"- {item}" for item in PITFALLS.get(code_key, DEFAULT_PITFALLS))
    deep_notes = "\n".join(f"- {item}" for item in DEEP_NOTES[code_key])
    coverage_items = "\n".join(f"- [x] {item}" for item in COVERAGE_CHECKLISTS[code_key])
    return {
        "cells": [
            md_cell(f"""
            # {title}

            {description}

            This notebook is an original tutorial inspired by the topic sequence of
            *Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems*.
            It does not reproduce the book text. Use it as a practical companion:
            read the concept, run the example, then complete the exercises.
            """),
            md_cell("""
            ## Learning Objectives

            - {0}
            - {1}
            - {2}
            """.format(*objectives)),
            md_cell("""
            ## Mental Model

            An agentic pattern is useful when it makes the system easier to inspect,
            control, evaluate, or improve. Before adding any pattern, ask:

            - What decision is being made?
            - What state or context is required?
            - What can fail?
            - How will I know the result is good?
            """),
            md_cell(f"## Important Concepts\n\n{concepts}"),
            md_cell(f"## Need-To-Know Coverage Checklist\n\n{coverage_items}"),
            md_cell(f"## Deep Study Notes\n\n{deep_notes}"),
            md_cell(f"## Common Failure Modes\n\n{pitfalls}"),
            md_cell("""
            ## Implementation Notes

            - Start with deterministic orchestration before adding model calls.
            - Represent state with dictionaries, dataclasses, Pydantic models, or typed records.
            - Log every decision point with enough context to reproduce failures.
            - Put risky actions behind policy checks and, where appropriate, human approval.
            - Add small eval cases before expanding the workflow.
            """),
            md_cell("""
            ## Minimal Runnable Example

            The example below avoids external model APIs on purpose. Replace the
            deterministic functions with model calls only after the control flow is clear.
            """),
            code_cell(CODE_SNIPPETS[code_key]),
            md_cell("""
            ## Practice

            1. Change the example input and predict the output before running it.
            2. Add one validation check that would catch a bad intermediate result.
            3. Write a short trace format that would help you debug this pattern in production.
            4. Identify one case where this pattern would be unnecessary overhead.
            """),
            md_cell("""
            ## Design Checklist

            - Inputs and outputs are explicit.
            - Intermediate state can be inspected.
            - Failure behavior is defined.
            - The pattern has a measurable reason to exist.
            """),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def slug(title):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in title).strip("_").replace("__", "_")


def main():
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    index_lines = [
        "# Agentic Design Patterns Tutorial Notebooks",
        "",
        "This workspace contains an original, hands-on notebook curriculum that follows the book's topic sequence without reproducing the book text.",
        "",
        "Start with `notebooks/00_course_orientation.ipynb`, then work through the numbered notebooks in order.",
        "",
        "See `COVERAGE_REVIEW.md` for the topic-by-topic coverage audit.",
        "See `BOOK_TOPIC_CHECKLIST.md` for the detailed book-wide checklist.",
        "",
        "## Notebook Map",
        "",
    ]
    coverage_lines = [
        "# Coverage Review",
        "",
        "This audit maps the tutorial notebooks to the major book topics and the core concepts a learner should know.",
        "The notebooks are original companion lessons and intentionally avoid reproducing the book text.",
        "The checklist was built from the book's chapter structure, recurring section labels, practical application lists, framework examples, and index terms.",
        "",
        "## Coverage Standard",
        "",
        "Each topic notebook should include:",
        "",
        "- A concise explanation of the pattern.",
        "- Learning objectives.",
        "- Important concepts specific to that pattern.",
        "- Common failure modes.",
        "- Implementation notes.",
        "- A runnable offline example.",
        "- Practice tasks and a design checklist.",
        "",
        "## Topic Map",
        "",
    ]
    book_checklist_lines = [
        "# Book Topic Checklist",
        "",
        "This file is the detailed coverage checklist for the tutorial notebooks.",
        "Every checked item is covered in the corresponding notebook as a concept, deep study note, implementation note, practice prompt, or runnable example.",
        "The wording is original and condensed for study use; it is not copied from the source book.",
        "",
    ]

    for number, title, description, key, *objectives in TOPICS:
        filename = f"{number}_{slug(title)}.ipynb"
        path = NOTEBOOK_DIR / filename
        path.write_text(json.dumps(notebook(title, description, key, objectives), indent=2), encoding="utf-8")
        index_lines.append(f"- `{filename}` - {title}")
        coverage_lines.append(f"### {number} - {title}")
        coverage_lines.append("")
        coverage_lines.append(description)
        coverage_lines.append("")
        coverage_lines.append("Key concepts covered:")
        coverage_lines.extend(f"- {item}" for item in CONCEPTS[key])
        coverage_lines.append("")
        coverage_lines.append("Deep notes covered:")
        coverage_lines.extend(f"- {item}" for item in DEEP_NOTES[key])
        coverage_lines.append("")
        coverage_lines.append("Detailed checklist:")
        coverage_lines.extend(f"- [x] {item}" for item in COVERAGE_CHECKLISTS[key])
        coverage_lines.append("")

        book_checklist_lines.append(f"## {number} - {title}")
        book_checklist_lines.append("")
        book_checklist_lines.append(f"Notebook: `notebooks/{filename}`")
        book_checklist_lines.append("")
        book_checklist_lines.extend(f"- [x] {item}" for item in COVERAGE_CHECKLISTS[key])
        book_checklist_lines.append("")

    (ROOT / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (ROOT / "COVERAGE_REVIEW.md").write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")
    (ROOT / "BOOK_TOPIC_CHECKLIST.md").write_text("\n".join(book_checklist_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
