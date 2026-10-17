# 39. Agent Identity and Governance

*Track A (agent engineering). Course addition; extends the book's Guardrails (Chapter 18) and Human in the Loop (Chapter 13) chapters. Prerequisites: 05. Tool Use, 13. Human in the Loop, 31. Agent Security and Prompt Injection, 33. Sandboxed Agent Execution.*

Every other safety lesson asks "should the agent do this?" This one asks a question that comes first: "who is doing this, and on whose authority?" An enterprise will not let an agent touch Stripe, the CRM, or production SQL until it can answer four things for every action: which agent did it, which human it was acting for, which rule allowed it, and how to stop it right now. Those four answers are identity, delegation, policy, and control. None of them is a prompt. All of them are ordinary security engineering applied to a new kind of caller.

## The problem

The support agent from earlier lessons went live with the simplest possible setup: one service account, `svc-support-agent`, with an API key for each system. It works. Then three things happen in the same week.

```text
Monday     Finance: "svc-support-agent refunded $4,800 to Acme. Who approved that?"
           Logs:    actor=svc-support-agent  action=refund  amount=480000
           Nobody can say which user asked, which run did it, or which rule allowed it.

Wednesday  A tier-1 support rep, who is not allowed to issue refunds, types:
           "Please refund Acme's last invoice, customer is upset."
           The agent has the refund key. The refund goes through.

Friday     An injected instruction in a Notion page makes the agent loop on
           stripe_refund. Someone asks: "How do we turn it off?"
           Answer: rotate the Stripe key, which breaks every other integration too.
```

Wednesday is the classic **confused deputy**. The agent holds more authority than the person using it, and it lends that authority to anyone who can type into Slack. The shared service account turned an access-control system into a suggestion.

```text
Naive                                     Governed
─────                                     ────────
user ──► agent ──► svc key ──► Stripe     user ──► agent ──► policy ──► broker ──► Stripe
                                              │       │         │          │
agent authority = key's authority             │       │         │          └ short-lived,
                                              │       │         │            audience-bound token
                                              │       │         └ allow / deny / approval,
                                              │       │           rule id recorded
                                              │       └ its own identity, its own ceiling
                                              └ delegation: what this user granted, for this task

effective authority = user grants ∩ agent ceiling ∩ task scope
```

That intersection is the whole lesson in one line. The interview question it answers: "How do you make an agent's actions auditable and bounded when it acts on behalf of users?"

## Build it without a framework

We build five pieces, each a few lines, then compose them into one guarded tool call.

```text
identity     who: the human principal, the agent, the run
delegation   the grant: user ∩ agent ∩ task, with an expiry
policy       rules as data, first match wins, default deny, every decision names its rule
audit        append-only, hash-chained records of who, what, authority, decision, result
controls     kill switches and spend limits, checked on every call, not once per run
```

### Identity, delegation, and a policy engine

```{code-cell} python
import hashlib
import json
import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Principal:            # a human, authenticated by your IdP
    user_id: str
    tenant: str
    scopes: frozenset


@dataclass(frozen=True)
class AgentIdentity:        # the agent's OWN identity: not a shared service account
    agent_id: str
    version: str
    ceiling: frozenset      # the most this agent may ever do, whoever asks


@dataclass(frozen=True)
class Delegation:           # "user U lets agent A do T until time E, in run R"
    user: Principal
    agent: AgentIdentity
    task_scopes: frozenset
    run_id: str
    expires_at: float

    def effective_scopes(self) -> frozenset:
        return self.user.scopes & self.agent.ceiling & self.task_scopes


@dataclass(frozen=True)
class Decision:
    effect: str             # "allow" | "deny" | "require_approval"
    rule: str               # which rule decided: the audit needs this, not just the effect
    reason: str


TOOLS = {                   # tool -> required scope, and whether it spends money
    "crm_lookup":    {"scope": "crm:read",        "spend_field": None},
    "sql_query":     {"scope": "sql:read",        "spend_field": None},
    "slack_post":    {"scope": "slack:write",     "spend_field": None},
    "stripe_refund": {"scope": "payments:refund", "spend_field": "amount_cents"},
}


def rule_expired(d, tool, args, now):
    if now >= d.expires_at:
        return Decision("deny", "R1-expiry", "delegation expired")


def rule_unknown_tool(d, tool, args, now):
    if tool not in TOOLS:
        return Decision("deny", "R2-registry", f"{tool} is not a registered tool")


def rule_scope(d, tool, args, now):
    need = TOOLS[tool]["scope"]
    if need not in d.effective_scopes():
        missing_from = [n for n, s in [("user", d.user.scopes), ("agent", d.agent.ceiling),
                                       ("task", d.task_scopes)] if need not in s]
        return Decision("deny", "R3-scope", f"{need} missing from {missing_from}")


def rule_refund_threshold(d, tool, args, now):
    if tool == "stripe_refund" and args.get("amount_cents", 0) > 50_000:
        return Decision("require_approval", "R4-refund>500", "refunds over $500 need a human")


RULES = [rule_expired, rule_unknown_tool, rule_scope, rule_refund_threshold]


def decide(d: Delegation, tool: str, args: dict, now: float) -> Decision:
    for rule in RULES:              # order matters: cheap, absolute denials first
        decision = rule(d, tool, args, now)
        if decision:
            return decision
    return Decision("allow", "R0-default-allow-within-scope", "scope satisfied")


support_agent = AgentIdentity("support-agent@acme", "2026.09.1",
                              frozenset({"crm:read", "sql:read", "slack:write", "payments:refund"}))
research_agent = AgentIdentity("research-subagent@acme", "2026.09.1",
                               frozenset({"crm:read", "sql:read"}))       # least privilege per agent
tier1 = Principal("dana", "acme", frozenset({"crm:read", "slack:write"}))
tier2 = Principal("lee", "acme", frozenset({"crm:read", "sql:read", "slack:write", "payments:refund"}))
task = frozenset({"crm:read", "slack:write", "payments:refund"})          # "resolve billing complaint"

cases = {
    "tier1 asks for refund (confused deputy)":
        decide(Delegation(tier1, support_agent, task, "run-1", 1000), "stripe_refund",
               {"amount_cents": 12_000}, now=10),
    "tier2 refund $120":
        decide(Delegation(tier2, support_agent, task, "run-2", 1000), "stripe_refund",
               {"amount_cents": 12_000}, now=10),
    "tier2 refund $900":
        decide(Delegation(tier2, support_agent, task, "run-2", 1000), "stripe_refund",
               {"amount_cents": 90_000}, now=10),
    "research subagent tries refund for tier2":
        decide(Delegation(tier2, research_agent, task, "run-3", 1000), "stripe_refund",
               {"amount_cents": 100}, now=10),
    "task has no sql scope":
        decide(Delegation(tier2, support_agent, task, "run-2", 1000), "sql_query", {}, now=10),
    "delegation expired":
        decide(Delegation(tier2, support_agent, task, "run-2", 1000), "crm_lookup", {}, now=2000),
}
for name, d in cases.items():
    print(f"{name:42s} {d.effect:17s} {d.rule:30s} {d.reason}")

assert cases["tier1 asks for refund (confused deputy)"].effect == "deny"
assert "user" in cases["tier1 asks for refund (confused deputy)"].reason
assert cases["tier2 refund $120"].effect == "allow"
assert cases["tier2 refund $900"].effect == "require_approval"
assert "agent" in cases["research subagent tries refund for tier2"].reason
assert "task" in cases["task has no sql scope"].reason
assert cases["delegation expired"].rule == "R1-expiry"
```

Look at the `reason` column. Each denial says *which* of the three grants was missing. The tier-1 rep lacked the scope, so the agent could not lend its own. The research subagent lacked it, so even a tier-2 user cannot route a refund through it. The task was "resolve a billing complaint," so SQL is out even though both user and agent could run SQL. Three independent ceilings, and the tightest one wins.

Three design decisions are worth saying out loud in an interview:

- **Rules are data with ids.** "Denied" is useless in an incident review. "Denied by R3-scope, missing from user" is actionable.
- **Default deny is structural.** An unregistered tool fails at R2 before any rule can accidentally allow it. The only allow path is "scope satisfied and nothing more specific objected."
- **The engine is pure.** `decide` takes the delegation, the call, and the time. No I/O. That makes it testable offline and lets you replay yesterday's audit log against today's rules to see what would change.

This is the classic split between a **policy decision point** (the `decide` function) and a **policy enforcement point** (the code that calls it before executing the tool). Production systems often put the decision point in a dedicated engine such as OPA or Cedar so security teams can own rules separately from agent code. The shape is identical.

### Audit, kill switches, spend limits, and the guarded call

```{code-cell} python
class AuditLog:
    """Append-only and hash-chained: editing or deleting any record breaks every later hash."""

    def __init__(self):
        self.records = []
        self._lock = threading.Lock()   # read-prev-then-append must be atomic under parallel tools

    def append(self, **record) -> dict:
        with self._lock:
            prev = self.records[-1]["hash"] if self.records else "genesis"
            body = json.dumps({**record, "prev": prev}, sort_keys=True)
            entry = {**record, "prev": prev, "hash": hashlib.sha256(body.encode()).hexdigest()}
            self.records.append(entry)
            return entry

    def verify(self) -> bool:
        prev = "genesis"
        for r in self.records:
            body = {k: v for k, v in r.items() if k != "hash"}
            if r["prev"] != prev or hashlib.sha256(
                    json.dumps(body, sort_keys=True).encode()).hexdigest() != r["hash"]:
                return False
            prev = r["hash"]
        return True


class KillSwitches:
    """Checked on EVERY call. A switch that is read once at run start is not a kill switch."""

    def __init__(self):
        self.off = set()     # entries like "global", "agent:support-agent@acme", "tool:stripe_refund"

    def blocked(self, d: Delegation, tool: str):
        for key in ("global", f"agent:{d.agent.agent_id}", f"tool:{tool}", f"tenant:{d.user.tenant}"):
            if key in self.off:
                return key


class SpendLedger:
    """Reserve before the side effect, like a card authorization; caps per run and per user/day."""

    def __init__(self, per_run: int, per_user_day: int):
        self.per_run, self.per_user_day = per_run, per_user_day
        self.by_run, self.by_user = {}, {}
        self._lock = threading.Lock()     # parallel tool calls must not both pass the check

    def reserve(self, d: Delegation, cents: int):
        with self._lock:
            run_total = self.by_run.get(d.run_id, 0) + cents
            user_total = self.by_user.get(d.user.user_id, 0) + cents
            if run_total > self.per_run:
                return f"run cap {self.per_run} exceeded"
            if user_total > self.per_user_day:
                return f"user daily cap {self.per_user_day} exceeded"
            self.by_run[d.run_id], self.by_user[d.user.user_id] = run_total, user_total

    def release(self, d: Delegation, cents: int):
        with self._lock:
            self.by_run[d.run_id] -= cents
            self.by_user[d.user.user_id] -= cents


def redact(args: dict) -> dict:
    # Audit what was done without copying customer data into a second system.
    return {k: (v if isinstance(v, int) else hashlib.sha256(str(v).encode()).hexdigest()[:12])
            for k, v in args.items()}


def guarded_call(d, tool_name, args, impls, *, audit, switches, ledger, now):
    """The enforcement point. Order: kill switch -> policy -> spend -> execute -> audit."""
    base = {"t": now, "run": d.run_id, "user": d.user.user_id, "agent": d.agent.agent_id,
            "agent_version": d.agent.version, "tool": tool_name, "args": redact(args)}
    if (switch := switches.blocked(d, tool_name)):
        audit.append(**base, effect="deny", rule="K-" + switch, result=None)
        return f"denied: kill switch {switch} is engaged"
    decision = decide(d, tool_name, args, now)
    if decision.effect != "allow":
        audit.append(**base, effect=decision.effect, rule=decision.rule, result=None)
        return f"{decision.effect}: {decision.reason} [{decision.rule}]"
    cents = args.get(TOOLS[tool_name]["spend_field"] or "", 0)
    if cents and (why := ledger.reserve(d, cents)):
        audit.append(**base, effect="deny", rule="S-spend", result=why)
        return f"denied: {why}"
    try:
        result = impls[tool_name](**args)
    except Exception as exc:
        if cents:
            ledger.release(d, cents)   # only if the provider confirms nothing happened
        audit.append(**base, effect="allow", rule=decision.rule, result=f"error:{exc}")
        return f"error: {exc}"
    audit.append(**base, effect="allow", rule=decision.rule, result="ok")
    return result


IMPLS = {
    "crm_lookup": lambda account: f"{account}: plan=Enterprise, owner=lee",
    "stripe_refund": lambda charge_id, amount_cents: f"refunded {amount_cents} on {charge_id}",
    "slack_post": lambda channel, text: f"posted to {channel}",
    "sql_query": lambda q: "rows=0",
}
audit, switches = AuditLog(), KillSwitches()
ledger = SpendLedger(per_run=30_000, per_user_day=100_000)
d = Delegation(tier2, support_agent, task, "run-7", expires_at=1000)
kw = dict(impls=IMPLS, audit=audit, switches=switches, ledger=ledger)

results = [
    guarded_call(d, "crm_lookup", {"account": "Acme"}, now=1, **kw),
    guarded_call(d, "stripe_refund", {"charge_id": "ch_1", "amount_cents": 20_000}, now=2, **kw),
    guarded_call(d, "stripe_refund", {"charge_id": "ch_2", "amount_cents": 20_000}, now=3, **kw),
]
switches.off.add("tool:stripe_refund")          # on-call engineer pulls the switch mid-run
results.append(guarded_call(d, "stripe_refund", {"charge_id": "ch_3", "amount_cents": 500}, now=4, **kw))
results.append(guarded_call(d, "slack_post", {"channel": "#billing", "text": "done"}, now=5, **kw))

for r in results:
    print("->", r)
for r in audit.records:
    print(r["t"], r["agent"], r["user"], r["tool"], r["effect"], r["rule"], r["hash"][:10])

assert results[1].startswith("refunded") and "run cap" in results[2]
assert "tool:stripe_refund" in results[3] and results[4] == "posted to #billing"
assert audit.verify()
audit.records[1]["args"]["amount_cents"] = 200     # an insider "fixes" the refund amount
assert not audit.verify()                           # tampering is detectable
print("tamper detected:", not audit.verify())
```

Four details carry most of the value:

1. **Every record names the user, the agent, its version, the run, and the rule.** Monday's question ("who approved the $4,800 refund?") becomes a query.
2. **The kill switch is scoped.** Turning off `tool:stripe_refund` stopped refunds and left Slack working. Friday's answer is no longer "rotate the key and break everything." Keep switches at global, tenant, agent, and tool granularity, and read them on every call from a fast shared store.
3. **Spend is reserved before the side effect.** Checking after the refund is reporting, not control. The lock matters because 00. What an agent actually is showed that `ToolNode` runs a turn's tool calls in parallel; two $200 refunds must not both see "$0 spent so far."
4. **Hash chaining makes tampering evident, not impossible.** Ship records to write-once storage (object lock, a separate account) so the chain is anchored somewhere the agent's operators cannot edit.

## How LangGraph does it

LangGraph and LangChain do not provide an identity provider or a policy engine. They provide the two hooks you need to attach yours: a typed **runtime context** that carries the delegation into every step without putting it in the prompt, and **tool-call middleware** that wraps every tool execution. The middleware is the enforcement point; `decide` stays the decision point.

```text
from scratch                         LangChain / LangGraph
────────────                         ─────────────────────
Delegation passed as an argument     context=Delegation(...) on invoke; read as request.runtime.context
guarded_call(d, tool, args, ...)     @wrap_tool_call middleware around every tool execution
denial string                        ToolMessage(status="error") the model can read and route around
require_approval                     interrupt() / HumanInTheLoopMiddleware (13. Human in the Loop)
audit records                        your AuditLog, plus the trace (32. Observability and Tracing)
```

Why context and not state or the prompt? State is visible to and updatable by nodes, and the prompt is visible to the model. Identity must be neither. The model never sees who it is acting for in a form it can rewrite; it only sees the consequences as tool results. The scripted model below stands in for a real model so the control flow is the object of study.

```{code-cell} python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedChatModel(GenericFakeChatModel):
    """Replays scripted turns so the control flow, not the model, is under study."""

    def bind_tools(self, tools, **kwargs):
        return self  # a real model would attach tool schemas here


def scripted(*turns):
    """Each turn is either final text or a list of (tool_name, args) tool calls."""
    messages = []
    for turn in turns:
        if isinstance(turn, str):
            messages.append(AIMessage(content=turn))
        else:
            messages.append(AIMessage(content="", tool_calls=[
                {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}
                for i, (name, args) in enumerate(turn)]))
    return ScriptedChatModel(messages=iter(messages))


from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


@tool
def crm_lookup(account: str) -> str:
    """Look up an account in the CRM by company name."""
    return IMPLS["crm_lookup"](account)


@tool
def stripe_refund(charge_id: str, amount_cents: int) -> str:
    """Refund part or all of a Stripe charge. Amount in cents."""
    return IMPLS["stripe_refund"](charge_id, amount_cents)


gov_audit, gov_switches = AuditLog(), KillSwitches()
gov_ledger = SpendLedger(per_run=100_000, per_user_day=200_000)
CLOCK = iter(range(100, 10_000))


@wrap_tool_call
def governance(request, handler):
    d: Delegation = request.runtime.context       # identity rides in context, not in the prompt
    call = request.tool_call
    outcome = {}

    def execute(**_):                               # the real execution is ToolNode's handler
        outcome["msg"] = handler(request)
        return outcome["msg"].content

    impls = {call["name"]: execute}
    text = guarded_call(d, call["name"], call["args"], impls, audit=gov_audit,
                        switches=gov_switches, ledger=gov_ledger, now=next(CLOCK))
    if "msg" in outcome:
        return outcome["msg"]
    return ToolMessage(content=text, tool_call_id=call["id"], name=call["name"], status="error")


def make_agent(model):
    return create_agent(model, tools=[crm_lookup, stripe_refund],
                        middleware=[governance], context_schema=Delegation)


ask = {"messages": [HumanMessage("Acme was double charged. Refund ch_9 ($120) and ch_8 ($900).")]}

tier1_run = make_agent(scripted(
    [("crm_lookup", {"account": "Acme"})],
    [("stripe_refund", {"charge_id": "ch_9", "amount_cents": 12_000})],
    "I can't issue refunds for you. I've flagged this for a billing lead.",
)).invoke(ask, context=Delegation(tier1, support_agent, task, "run-t1", 10_000))

tier2_run = make_agent(scripted(
    [("stripe_refund", {"charge_id": "ch_9", "amount_cents": 12_000})],
    [("stripe_refund", {"charge_id": "ch_8", "amount_cents": 90_000})],
    "Refunded $120 on ch_9. The $900 refund on ch_8 is waiting for approval.",
)).invoke(ask, context=Delegation(tier2, support_agent, task, "run-t2", 10_000))


def tool_texts(result):
    return [m.content for m in result["messages"] if isinstance(m, ToolMessage)]


print("tier1:", tool_texts(tier1_run))
print("tier2:", tool_texts(tier2_run))
for r in gov_audit.records:
    print(r["run"], r["user"], r["tool"], r["effect"], r["rule"])

assert "payments:refund missing from ['user']" in tool_texts(tier1_run)[1]
assert tool_texts(tier2_run)[0] == "refunded 12000 on ch_9"
assert tool_texts(tier2_run)[1].startswith("require_approval")
assert [r["effect"] for r in gov_audit.records] == ["allow", "deny", "allow", "require_approval"]
assert gov_audit.verify()
```

Map it back. `request.runtime.context` is the `Delegation` we passed by hand before. The `governance` middleware *is* `guarded_call`: it runs the kill switch, the policy, and the spend reservation, and only then calls `handler(request)`, which is the real tool execution inside `ToolNode`. A denial becomes a `ToolMessage` with `status="error"`, so the model learns the action failed and why, and can tell the user instead of hallucinating success. The tier-1 run is Wednesday's scenario, fixed: the agent had the refund scope, but the user did not, so the intersection was empty.

In production the `require_approval` branch would call `interrupt()` with the proposed action, pause the graph on a checkpointer, and resume with the approver's decision (13. Human in the Loop). `HumanInTheLoopMiddleware` packages that pattern per tool. Keep the approver's identity in the audit record too: an approval is itself an authorization decision.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model

real = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
live_agent = create_agent(
    real, tools=[crm_lookup, stripe_refund], middleware=[governance], context_schema=Delegation,
    system_prompt="You are Acme's support agent. If a tool is denied, explain why and stop.")
out = live_agent.invoke(ask, context=Delegation(tier1, support_agent, task, "run-live", 10_000))
print(out["messages"][-1].content)
for r in gov_audit.records[-3:]:
    print(r["user"], r["tool"], r["effect"], r["rule"])
```

### Where MCP authorization fits

When the tools come from remote MCP servers (10. Model Context Protocol), part of this machinery moves onto the wire. The MCP client in your host obtains OAuth tokens for each server, and the server enforces scopes. The `2026-07-28` revision of the MCP specification hardened exactly the parts that matter for agents acting on behalf of users:

```text
Issuer validation (RFC 9207)     the client checks the `iss` in the authorization response,
                                 which defeats mix-up attacks where a malicious server tricks
                                 the client into sending a code to the wrong authorization server
Credentials bound to issuer      a token obtained from one authorization server is not
                                 presented elsewhere; confused-deputy by token replay is closed
CIMD replaces DCR                clients identify themselves with a Client ID Metadata Document
                                 (a URL you control) instead of anonymous Dynamic Client
                                 Registration, which is now deprecated
Enterprise Managed Auth          an official extension for IdP-administered access, so an
                                 enterprise admin, not each user, decides which servers are allowed
Mcp-Method / Mcp-Name headers    gateways can route and authorize requests per method and tool
                                 without parsing JSON bodies
```

Your in-process policy engine does not go away. MCP authorization answers "may this client call this server with these scopes?" Your policy answers "may this agent, for this user, in this task, make this particular call with these arguments right now?" You need both. The MCP roadmap lists agent identity (DPoP-bound tokens and workload identity federation) as future work, so treat per-agent identity at the protocol level as a direction, not a shipped feature.

## What goes wrong in production

1. **Shared service accounts.** Symptom: audit logs say "svc-agent did it" and nothing else. Cause: one credential for every agent and every user. Fix: a distinct identity per agent (and per version), the user principal carried through delegation, and the run id on every record. If a downstream API only supports one key, the broker is where attribution is added.

2. **Confused deputy.** Symptom: users obtain actions through the agent that they could not perform directly. Cause: effective authority is the agent's authority. Fix: intersect user grants, agent ceiling, and task scope on every call, and prefer on-behalf-of tokens so the downstream system also sees the user.

3. **Long-lived, broad tokens.** Symptom: a leaked key keeps working for months and can do everything. Cause: static API keys in configuration. Fix: short-lived tokens minted per run or per call, bound to one audience, with the narrowest scopes. Expiry is a passive kill switch that fires even when nobody notices the incident.

4. **Kill switch checked once.** Symptom: the switch is flipped and the running agent keeps refunding for twenty minutes. Cause: the flag was read at run start and cached. Fix: check on every tool call against a low-latency shared store, and test the switch in game days the way you test failover.

5. **Spend limits enforced after the fact.** Symptom: a loop issues forty refunds before the nightly report notices. Cause: spend is summed from logs. Fix: reserve before executing, under a lock or an atomic counter, with per-run, per-user, per-tenant, and per-day caps. Treat model token spend the same way (16. Resource-Aware Optimization).

6. **Audit logs that leak or lie.** Symptom: the audit table becomes the largest store of customer PII, or someone quietly edits a record. Cause: raw arguments logged into mutable storage. Fix: redact or hash sensitive arguments, keep references to source records, hash-chain entries, and ship them to write-once storage owned by a different team.

7. **Policy in the prompt.** Symptom: "only refund if the user is a billing lead" in the system prompt, and an injected Notion page says the user is a billing lead. Cause: the model enforces nothing. Fix: the prompt may *explain* policy so the model plans well, but enforcement lives in code the model cannot reach.

## What changed by 2026

- **Agents are treated as their own principals.** The practice moved from "the agent uses our service account" to "each agent is a registered workload with an owner, a version, a ceiling of permissions, and an audit trail," with the human's delegation carried separately. Interviewers now expect the intersection model and the confused-deputy vocabulary.
- **MCP authorization hardened.** The `2026-07-28` MCP revision added issuer validation per RFC 9207, bound credentials to the issuing authorization server, deprecated Dynamic Client Registration in favor of Client ID Metadata Documents, and shipped Enterprise Managed Authorization as an official extension. Agent-specific identity (DPoP, workload identity federation) is on the published roadmap, not yet in the spec.
- **Harnesses expose enforcement hooks.** `create_agent` middleware (`wrap_tool_call`, `HumanInTheLoopMiddleware`, `ToolCallLimitMiddleware`, `ModelCallLimitMiddleware`) and typed runtime context make the enforcement point a first-class, testable component instead of code scattered through tools. Deep Agents adds path-level filesystem permissions and per-tool `interrupt_on` approvals. The OpenAI Agents SDK has guardrails and MCP tool-approval callbacks.
- **Credentials moved out of the agent's reach.** Following the brain/hands separation in 33. Sandboxed Agent Execution, credentials sit with a broker on the trusted side and are attached per request, so a compromised model or sandbox cannot exfiltrate them.
- **Governance became operational, not documentary.** Kill switches, spend caps, and approval queues are part of the on-call runbook and are exercised, not just written into a responsible-AI policy.

## Interview questions

**Q1. How do you make an agent's actions auditable?** (reported at Palantir)

Strong: Every tool call passes through one enforcement point that writes an audit record before and after the side effect. The record names the human principal, the agent identity and version, the run id, the tool, redacted arguments, the policy decision with the rule id that made it, any approver, and the result. Records are append-only and hash-chained, and shipped to write-once storage owned by a different team, so tampering is evident. I also keep the trace for the reasoning context, but the audit record is the authoritative, structured source. Monday-morning questions like "who approved this refund" become a query, not an investigation.

Weak: "We log all the LLM calls."

**Q2. What is the confused-deputy problem for agents, and how do you prevent it?**

Strong: The agent holds more authority than the user talking to it, so any user can borrow the agent's authority by asking. A support rep who cannot issue refunds asks the agent, which can, and the refund happens. The fix is to compute effective authority on every call as the intersection of the user's grants, the agent's own ceiling, and the scope of the current task, and to use on-behalf-of tokens so downstream systems see the user too. Prompt instructions do not solve it because the model cannot verify who the user is.

Weak: "Tell the agent to check the user's role."

**Q3. Should each agent have its own identity? Why not one service account?**

Strong: Yes. A distinct identity per agent, and ideally per version, gives least privilege, since a research subagent never needs refund scope. It gives attribution, since logs say which agent acted. And it gives targeted control, since I can revoke or kill one agent without breaking every integration. A shared service account collapses all of that: its permissions are the union of every agent's needs, and revoking it is an outage.

Weak: "One account is simpler to manage."

**Q4. Design the kill switch for a production agent.**

Strong: Switches at several scopes, global, tenant, agent, agent version, and tool, stored in a low-latency shared store and checked on every tool call, not at run start. Engaging one makes subsequent calls fail with a clear error the model reports to the user, while in-flight runs checkpoint so they can resume later. It needs an owner and an on-call runbook, it should be exercised in game days, and engaging it is itself audited. Short token lifetimes act as a backstop kill switch if the control plane is unavailable.

Weak: "We can shut down the server."

**Q5. How do you enforce spend limits on an agent that can move money?**

Strong: Reserve before executing, like a card authorization: check the requested amount against per-run, per-user, per-tenant, and per-day caps atomically, reserve it, execute, then settle or release only if the provider confirms nothing happened. The check must be atomic because tool calls in one turn run in parallel. Above a threshold the policy returns require-approval rather than deny, routing to a human. Model token spend gets the same treatment with its own budget.

Weak: "We monitor spend in a dashboard."

**Q6. Where should policy live: in the prompt, in the tools, or somewhere else?**

Strong: Enforcement lives in a policy decision point the model cannot reach, called from a single enforcement point around tool execution. Rules are data with ids so decisions are explainable and replayable, the default is deny, and the engine is pure so it can be unit-tested and diffed against historical traffic. The prompt can describe the policy so the model plans sensibly and does not waste calls, but it enforces nothing. Individual tools still validate their own arguments, as defense in depth.

Weak: "In the system prompt, with strong wording."

**Q7. What did the 2026 MCP authorization changes fix, and what do they not cover?**

Strong: The `2026-07-28` revision requires issuer validation per RFC 9207, which stops mix-up attacks, binds credentials to the authorization server that issued them, which stops token replay across servers, deprecates anonymous Dynamic Client Registration in favor of Client ID Metadata Documents, and adds Enterprise Managed Authorization as an official extension so admins control which servers are allowed. What it does not cover is per-call business policy: whether this agent, for this user, may refund this amount right now. That stays in my policy engine. Per-agent identity at the protocol level is on the roadmap, not in the spec yet.

Weak: "MCP handles auth, so we're covered."

## Build it yourself

Extend the governance layer from this lesson:

1. Replace the `require_approval` string with a real pause: in the `governance` middleware, call `interrupt()` with the proposed action, compile the agent with an `InMemorySaver`, and resume with `Command(resume={"approved": True, "approver": "maria"})`.
2. Record the approver in the audit record, and add a rule that the approver must differ from the requesting user (separation of duties).
3. Add a per-tenant kill switch and a `tenant` check in policy: `crm_lookup` for an account whose tenant differs from the user's tenant is denied.
4. Add `replay(audit_records, rules)` that re-evaluates every past decision under a new rule set and returns the records whose effect would change.

Acceptance checks:

- The $900 refund pauses, and after resume with approval it executes exactly once; the audit shows `require_approval` then `allow` with `approver: maria`.
- If `lee` tries to approve `lee`'s own refund, the call is denied with a separation-of-duties rule id.
- Engaging `tenant:acme` denies every subsequent call for acme users while another tenant's run continues.
- Lowering the refund threshold to $100 and replaying the tier-2 run flags the $120 refund as "would now require approval."
- `audit.verify()` is still `True` after all runs.

## Key takeaways

- Every agent action needs four answers: which agent, for which human, under which rule, and how to stop it.
- Effective authority is the intersection of the user's grants, the agent's ceiling, and the task's scope, computed on every call; anything else is a confused deputy.
- Give each agent its own identity and least privilege; shared service accounts destroy attribution and targeted revocation.
- Keep the policy decision point pure and data-driven, and the enforcement point a single wrapper around tool execution.
- Audit records must be structured, redacted, hash-chained, and stored where operators cannot edit them.
- Kill switches and spend caps are checked on every call and reserved before side effects, not reported afterwards.
- MCP's 2026 authorization hardening secures the client-to-server hop; per-call business policy is still yours.
