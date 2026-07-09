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
            md_cell(f"""
            ## Important Concepts

            {concepts}
            """),
            md_cell(f"""
            ## Common Failure Modes

            {pitfalls}
            """),
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
        "",
        "## Notebook Map",
        "",
    ]
    coverage_lines = [
        "# Coverage Review",
        "",
        "This audit maps the tutorial notebooks to the major book topics and the core concepts a learner should know.",
        "The notebooks are original companion lessons and intentionally avoid reproducing the book text.",
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

    (ROOT / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (ROOT / "COVERAGE_REVIEW.md").write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
