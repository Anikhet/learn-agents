# H. Forward-Deployed Engineering: taking an agent from customer ask to production

*Track D (infra + interviews) with a Track A (agent engineering) flavor. Book: draws on Chapters 13 (Human in the Loop), 16 (Resource-Aware Optimization), 18 (Guardrails), and 19 (Evaluation and Monitoring). Prerequisites: D. Building an Agent, 19. Evaluation and Monitoring, 13. Human in the Loop, 39. Agent Identity, Authorization, and Governance.*

A forward-deployed engineer (FDE) is an engineer who sits with a customer and turns an ambiguous request ("we want an AI agent for support") into a system that runs in the customer's environment, on the customer's data, under the customer's rules, with numbers both sides signed. Palantir popularized the title. AI labs and agent companies adopted it, because agent deployments fail far more often on scoping, access, and sign-off than on model quality.

The technical content of this lesson is the part of agent engineering that no framework ships: discovery, a readiness checklist with evidence, success metrics with confidence intervals, an eval-gated rollout with customer sign-off, and a cost model. You will build each as code, because the FDE's most useful habit is turning a meeting into an artifact someone can check.

## The problem

Globex, a 900-person SaaS company, signs a pilot. The VP of Support says: "We want the agent from your demo. It should handle our tickets and issue refunds." Their stack is the course's running example: Slack for internal escalation, Notion for policies, Stripe for billing, a CRM, a SQL warehouse, and the web.

The naive FDE does what worked in the demo:

```text
Week 1   build a slick agent on a CSV export someone emailed; exec demo goes well
Week 2   ask for Stripe and CRM API keys                → security review: 6-8 weeks
Week 3   agent answers from a 2023 policy doc in Notion → nobody owns policy freshness
Week 6   "is it good enough?"                            → no baseline, no agreed metric
Week 9   first live refund goes wrong                    → no approval gate, no audit trail
Week 12  the champion changes jobs                        → no signed scope; project stalls
Week 16  finance sees the model bill                      → nobody modeled cost per ticket
```

Every one of those failures was decided in week 0, by what the FDE did not ask. The engagement needs a shape with gates, where each gate is evidence both sides can check:

```text
 DISCOVERY        SCOPE            ACCESS           PILOT (shadow)     CANARY          PRODUCTION
 who, what,   ─►  one workflow, ─► data + perms ─►  agent runs,    ─►  5% → 25%    ─►  GA + handoff
 baseline,        metric, target,  approved,        humans still       live, kill      runbook,
 data map         owner signs      scoped creds     act; compare       switch armed    on-call owner
     │                │                │                 │                  │               │
   GATE:            GATE:            GATE:             GATE:              GATE:           GATE:
 baseline         signed          least-privilege   eval sign-off     beats baseline   customer team
 measured         one-pager       + audit log       (thresholds met   with no sev-1    owns it; FDE
                                  verified          with CI)          incidents        exits
```

The FDE's job is to move the customer through that pipeline without skipping gates, and to stop the engagement honestly when a gate cannot be met.

## Build it without a framework

### Discovery: pick the workflow, not the vision

Discovery produces a list of candidate use cases. Each gets rough scores and, more importantly, hard gates that a high score cannot override. The rule that surprises customers: the pilot is usually not the most valuable idea. It is the most valuable idea that clears the gates, because a stalled pilot delivers zero.

```{code-cell} python
from dataclasses import dataclass

WEIGHTS = {"value": 0.35, "feasibility": 0.25, "data": 0.25, "safety": 0.15}


@dataclass
class UseCase:
    name: str
    value: int          # 1-5: business impact if it works
    feasibility: int    # 1-5: can current models do this reliably in this workflow?
    data: int           # 1-5: data exists, is accessible, and is permissioned
    risk: int           # 1-5: blast radius (5 = irreversible and customer-facing)
    owner: str          # named decision owner who feels the baseline metric ("" = none)
    baseline: str       # the measured number the agent must beat ("" = unmeasured)
    hitl_possible: bool

    def score(self):
        parts = {"value": self.value, "feasibility": self.feasibility, "data": self.data, "safety": 6 - self.risk}
        return round(sum(WEIGHTS[k] * v for k, v in parts.items()), 2)

    def blockers(self):
        checks = [("named decision owner", bool(self.owner)),
                  ("baseline measured", bool(self.baseline)),
                  ("data readiness >= 3", self.data >= 3),
                  ("risk >= 4 needs a human-in-the-loop step", self.risk < 4 or self.hitl_possible)]
        return [label for label, ok in checks if not ok]


CANDIDATES = [
    UseCase("Autonomous refunds (no human)", 5, 3, 3, 5, "VP Support", "$1.9M/yr refund ops", False),
    UseCase("Tier-1 billing questions, agent replies directly", 4, 3, 3, 3, "Dir. Support Ops",
            "median first response 3h40m; 31% of tickets", True),
    UseCase("Exec ask-anything BI chatbot", 3, 2, 2, 2, "", "", True),
    UseCase("Invoice-dispute triage with refund proposal (human approves)", 4, 4, 4, 4, "Dir. Support Ops",
            "11 min median handling; 7% reopen rate", True),
]

ranked = sorted(CANDIDATES, key=lambda u: u.score(), reverse=True)
for u in ranked:
    print(f"{u.score():4.2f}  {u.name:62s} blockers={u.blockers()}")
pilot = next(u for u in ranked if not u.blockers())
print("\npilot:", pilot.name)

assert CANDIDATES[0].blockers() == ["risk >= 4 needs a human-in-the-loop step"]   # highest value, blocked
assert "baseline measured" in CANDIDATES[2].blockers()
assert pilot.name.startswith("Invoice-dispute")                                # not the headline ask
assert pilot.risk < 5 and pilot.baseline
```

The VP asked for autonomous refunds. The FDE proposes invoice-dispute triage with a refund *proposal* a human approves, with tier-1 billing answers as the runner-up. Same systems, same value story, a gate the customer can live with, and a path to autonomy later when the approval data shows which refunds humans always approve. That conversation is most of the job.

### The one-page scope

The scope is short enough that a busy executive reads it and signs it. Everything in it is checkable.

```text
Workflow        Invoice-dispute tickets tagged "billing" in the helpdesk (about 1,100/month)
Agent does      Reads ticket, CRM account, Stripe invoice, Notion policy; drafts reply;
                proposes refund <= $5,000 with rationale
Human does      Approves or edits every refund; approves reply for first 2 weeks of canary
Out of scope    Refunds > $5,000, contract changes, anything outside "billing" tag
Data            Stripe read + refund (restricted key), CRM read, Notion "Policies" space read,
                helpdesk read/write drafts. No SQL warehouse in pilot.
Metric          Median handling time (baseline 11 min) and reopen rate (baseline 7%)
Target          Handling time <= 6 min; reopen rate not worse than baseline; zero sev-1
Owner           Director of Support Ops (signs eval results and go/no-go)
Kill criteria   Any sev-1, or targets not met by week 10 → stop or rescope
```

### The deployment-readiness checklist, scored with evidence

Most "readiness reviews" are a meeting where someone says "I think we're fine." Make it a checklist where every item needs *evidence* (a link, a ticket, a dashboard, a signed doc), some items are blocking, and the verdict is computed. The checklist below covers the eight areas a deployment fails on: scope, data access, permissions, evals, human oversight, operations, cost, and handoff.

```{code-cell} python
@dataclass
class Item:
    area: str
    check: str
    weight: int
    blocking: bool
    evidence: str = ""        # empty means "not done", no matter what anyone says in the meeting

    @property
    def done(self):
        return bool(self.evidence.strip())


def readiness(items, stage="production"):
    total = sum(i.weight for i in items)
    earned = sum(i.weight for i in items if i.done)
    blockers = [f"{i.area}: {i.check}" for i in items if i.blocking and not i.done]
    by_area = {}
    for i in items:
        e, t = by_area.get(i.area, (0, 0))
        by_area[i.area] = (e + (i.weight if i.done else 0), t + i.weight)
    pct = round(100 * earned / total)
    if blockers:
        verdict = "NOT READY"
    elif pct < 95:            # non-blocking gaps become written conditions with owners
        verdict = "READY WITH CONDITIONS"
    else:
        verdict = "READY"
    return {"stage": stage, "score": pct, "verdict": verdict, "blockers": blockers,
            "weakest_area": min(by_area, key=lambda a: by_area[a][0] / by_area[a][1])}


CHECKLIST = [
    Item("scope", "one-page scope signed by decision owner", 3, True, "SCOPE-v2.pdf signed 2026-08-04"),
    Item("scope", "baseline metric measured on >= 4 weeks of data", 3, True, "dash/support-baseline"),
    Item("data", "data processing agreement and residency confirmed", 3, True, "DPA-114"),
    Item("data", "retrieval corpus has an owner and freshness SLA", 2, False),
    Item("permissions", "agent runs as a service identity, least-privilege scopes", 3, True, "IAM-ticket-8812"),
    Item("permissions", "per-user permission checks on retrieval", 2, True, "test: perm_filter_suite green"),
    Item("permissions", "every tool call audit-logged with user, args, result", 3, True, "log schema v3 in SIEM"),
    Item("evals", "golden set >= 200 real tickets, labeled by customer SMEs", 3, True, "dataset globex-billing-v4"),
    Item("evals", "thresholds met with confidence interval, signed", 3, True),
    Item("evals", "regression suite runs on every prompt/model change", 2, False, "ci/agent-evals.yml"),
    Item("oversight", "refunds require human approval; approval UI tested", 3, True, "UAT-22 pass"),
    Item("oversight", "kill switch tested by customer on-call", 2, True),
    Item("operations", "on-call rotation and runbook owned by customer team", 3, True, "PagerDuty: support-ai"),
    Item("operations", "traces retained and searchable by ticket id", 2, False, "LangSmith project globex-prod"),
    Item("cost", "cost per ticket modeled and budget alert configured", 2, False, "sheet cost-model-v3"),
    Item("handoff", "customer engineers made one change end to end", 2, False),
]

report = readiness(CHECKLIST)
print(report)
assert report["verdict"] == "NOT READY"
assert report["blockers"] == ["evals: thresholds met with confidence interval, signed",
                              "oversight: kill switch tested by customer on-call"]

# Close the two blockers with evidence and re-score.
for i in CHECKLIST:
    if i.check.startswith("thresholds met"):
        i.evidence = "EVAL-SIGNOFF-3 (Dir. Support Ops)"
    if i.check.startswith("kill switch"):
        i.evidence = "drill 2026-09-02, 40s to disable"
after = readiness(CHECKLIST)
print(after)
assert after["verdict"] == "READY WITH CONDITIONS" and after["blockers"] == []
assert after["weakest_area"] in {"data", "handoff"}
```

Three properties matter. An item without evidence is not done, which moves the argument from opinion to artifacts. Blocking items cannot be averaged away: 94 percent readiness with an untested kill switch is "not ready." And the verdict names the weakest area, which is where the FDE spends next week.

### Success metrics and eval sign-off, with honest uncertainty

The pilot runs in **shadow mode**: the agent processes real tickets, humans still do the work, and the two are compared. The sign-off question is "did the agent meet the thresholds the owner signed?" The trap is small samples. Forty-three agreements out of fifty is 86 percent, and it is also consistent with a true rate well below the 85 percent target. Use a confidence interval and sign off on its lower bound.

```text
Wilson score interval for a proportion p̂ = k/n at z = 1.96 (95%):

  center = (p̂ + z²/2n) / (1 + z²/n)
  half   = z · sqrt(p̂(1−p̂)/n + z²/4n²) / (1 + z²/n)

Example: k = 43, n = 50  → p̂ = 0.86, interval ≈ [0.74, 0.93]
         k = 430, n = 500 → p̂ = 0.86, interval ≈ [0.83, 0.89]
Same point estimate; only the second one clears a 0.80 floor, and neither clears 0.85.
```

```{code-cell} python
import math
import random


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(center - half, 3), round(center + half, 3)


# Shadow-mode log: one record per ticket. Deterministic synthetic data stands in for the
# customer's real comparison export.
rng = random.Random(7)
shadow = [{"agrees_with_human": rng.random() < 0.92,
           "escalated": rng.random() < 0.08,
           "sev1": False,
           "handle_min": round(rng.uniform(3.0, 8.5), 1),
           "cost_usd": round(rng.uniform(0.04, 0.16), 3)} for _ in range(400)]

THRESHOLDS = {"agreement_lower_bound": 0.85, "escalation_max": 0.10, "sev1_max": 0,
              "median_handle_min": 6.0, "cost_per_ticket_max": 0.15}


def sign_off(log, t):
    n = len(log)
    k = sum(r["agrees_with_human"] for r in log)
    lo, hi = wilson(k, n)
    handle = sorted(r["handle_min"] for r in log)[n // 2]
    metrics = {"n": n, "agreement": round(k / n, 3), "agreement_ci": (lo, hi),
               "escalation": round(sum(r["escalated"] for r in log) / n, 3),
               "sev1": sum(r["sev1"] for r in log), "median_handle_min": handle,
               "cost_per_ticket": round(sum(r["cost_usd"] for r in log) / n, 3)}
    checks = {"agreement": lo >= t["agreement_lower_bound"],
              "escalation": metrics["escalation"] <= t["escalation_max"],
              "sev1": metrics["sev1"] <= t["sev1_max"],
              "handle_time": handle <= t["median_handle_min"],
              "cost": metrics["cost_per_ticket"] <= t["cost_per_ticket_max"]}
    return metrics, checks, all(checks.values())


m50, c50, ok50 = sign_off(shadow[:50], THRESHOLDS)
m400, c400, ok400 = sign_off(shadow, THRESHOLDS)
print("n=50: ", m50["agreement"], m50["agreement_ci"], "->", "SIGN" if ok50 else "NOT YET", c50)
print("n=400:", m400["agreement"], m400["agreement_ci"], "->", "SIGN" if ok400 else "NOT YET", c400)

assert wilson(43, 50) == (0.738, 0.93)
assert not c50["agreement"]                       # small sample: lower bound under the target
assert m400["agreement_ci"][1] - m400["agreement_ci"][0] < m50["agreement_ci"][1] - m50["agreement_ci"][0]
assert c400["sev1"] and c400["cost"]
```

Look at the n = 50 line. The point estimate may well sit above 85 percent, and the lower bound does not. Signing off at n = 50 is how pilots "succeed" and then disappoint in production. Settle the sample size in the scope document before the pilot starts, so nobody is tempted to stop the moment the number looks good.

Agreement with the human is only a proxy. Pair it with outcome metrics the business owns (handling time, reopen rate, CSAT) and a qualitative review of every disagreement. Some disagreements are the agent being right and the human being wrong. Those become labeled examples, not errors. See 22. Evaluation Reliability Deep Dive for grader agreement and 37. Trajectory Evals and Statistics for sample sizes.

## How LangGraph does it

The rollout is a long-running process with human decisions at gates, measured in weeks. That is exactly the shape LangGraph's checkpointer and `interrupt` were built for: each stage is a node, each gate is a conditional edge, and customer sign-off is an `interrupt` that can wait days and resume on a different machine.

```text
START ─► readiness ──NOT READY──► remediate ─► END (re-enter next week)
             │
           READY*
             ▼
          shadow ─► eval_gate ──fail──► rescope ─► END
                        │
                      pass
                        ▼
                  owner_signoff  (interrupt: waits for the decision owner)
                        │
               approve ─┴─ reject ──► rescope
                  ▼
                canary ─► END
```

```{code-cell} python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class Rollout(TypedDict, total=False):
    customer: str
    readiness: dict
    eval: dict
    signoff: dict
    stage: str
    audit: list


def audit(state, event):
    return (state.get("audit") or []) + [event]


def readiness_node(s: Rollout):
    r = readiness(CHECKLIST)
    return {"readiness": r, "audit": audit(s, f"readiness {r['verdict']} {r['score']}%")}


def shadow_node(s: Rollout):
    metrics, checks, ok = sign_off(shadow, THRESHOLDS)
    return {"eval": {"metrics": metrics, "checks": checks, "pass": ok},
            "audit": audit(s, f"shadow eval pass={ok}")}


def owner_signoff(s: Rollout):
    # Everything before interrupt() re-runs on resume, so this node only reads state.
    decision = interrupt({"ask": "Approve canary at 5% of billing tickets?",
                          "owner": "Dir. Support Ops", "evidence": s["eval"]["metrics"]})
    return {"signoff": decision, "audit": audit(s, f"signoff {decision['decision']} by {decision['by']}")}


def canary(s: Rollout):      return {"stage": "canary-5pct", "audit": audit(s, "canary started, kill switch armed")}
def remediate(s: Rollout):   return {"stage": "remediate", "audit": audit(s, f"blocked by {s['readiness']['blockers']}")}
def rescope(s: Rollout):     return {"stage": "rescope", "audit": audit(s, "gate failed; back to scoping")}


g = StateGraph(Rollout)
for name, fn in [("readiness", readiness_node), ("shadow", shadow_node), ("owner_signoff", owner_signoff),
                 ("canary", canary), ("remediate", remediate), ("rescope", rescope)]:
    g.add_node(name, fn)
g.add_edge(START, "readiness")
g.add_conditional_edges("readiness", lambda s: "remediate" if s["readiness"]["blockers"] else "shadow",
                        ["remediate", "shadow"])
g.add_conditional_edges("shadow", lambda s: "owner_signoff" if s["eval"]["pass"] else "rescope",
                        ["owner_signoff", "rescope"])
g.add_conditional_edges("owner_signoff", lambda s: "canary" if s["signoff"]["decision"] == "approve" else "rescope",
                        ["canary", "rescope"])
for n in ("canary", "remediate", "rescope"):
    g.add_edge(n, END)
rollout = g.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "globex-billing"}}
paused = rollout.invoke({"customer": "Globex"}, cfg)
ask = paused["__interrupt__"][0].value
print("waiting on:", ask["owner"], "|", ask["ask"])
print("state.next:", rollout.get_state(cfg).next)

# Days later, possibly another process: the owner approves in the approval UI.
done = rollout.invoke(Command(resume={"decision": "approve", "by": "Dir. Support Ops"}), cfg)
for line in done["audit"]:
    print("  audit:", line)

assert rollout.get_state(cfg).next == ()
assert done["stage"] == "canary-5pct"
assert done["audit"][-2] == "signoff approve by Dir. Support Ops"
assert [e.split()[0] for e in done["audit"]] == ["readiness", "shadow", "signoff", "canary"]
```

Map it back to the plain-Python pieces:

| Plain Python | LangGraph | Underneath |
|---|---|---|
| `readiness(CHECKLIST)` then `if verdict == ...` | `readiness` node + conditional edge | The router reads `readiness.blockers` from state after the node's super-step |
| `sign_off(shadow, THRESHOLDS)` | `shadow` node + conditional edge on `eval.pass` | Same function, now a step whose inputs and outputs are checkpointed |
| "Email the director and wait" | `interrupt({...})` in `owner_signoff` | State is saved under `thread_id`; the run returns with `__interrupt__`; nothing is holding a process open |
| Director replies a week later | `invoke(Command(resume=...), cfg)` | The checkpoint is loaded, the node re-runs from the top, and `interrupt()` returns the decision |
| Audit spreadsheet | `audit` list in state + checkpoint history | Every super-step is persisted; `get_state_history` is the record of who approved what, when |

In a real deployment the checkpointer is Postgres, the interrupt payload is rendered in an approval UI the customer's director logs into, and the resume carries the approver's identity from SSO, not a string. The graph shape is the same.

### A live helper: turning discovery notes into a draft scope

The FDE's raw material is interview notes. A model with structured output can draft the scope fields and, more usefully, list the questions the notes do not answer. The FDE edits; the owner signs.

```{code-cell} python live
import os
from pydantic import BaseModel
from langchain.chat_models import init_chat_model


class DraftScope(BaseModel):
    workflow: str
    agent_does: list[str]
    human_does: list[str]
    out_of_scope: list[str]
    baseline_metric: str
    open_questions: list[str]


notes = """VP Support wants refunds automated. Dir. Support Ops says billing disputes take ~11 min each,
about 1,100/month, reopen ~7%. Refund policy lives in Notion, last edited by someone who left.
Security needs 6 weeks for new API keys. Finance cares about refunds over $5k."""
model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
draft = model.with_structured_output(DraftScope).invoke(
    "Draft a pilot scope from these discovery notes. List what is missing as open questions.\n" + notes)
print(draft.model_dump_json(indent=2))
```

## What goes wrong in production

1. **Demo on exported data.** Symptom: great demo, then months of stall. Cause: the riskiest assumption (can we get governed access to live systems?) was tested last. Fix: start the access and security review in week 0, in parallel with discovery; build the first slice on the customer's real systems in their environment, even if it is ugly.

2. **No baseline.** Symptom: "is 78 percent good?" arguments with no answer. Cause: nobody measured the current process. Fix: baseline is a blocking checklist item; if the helpdesk cannot export handling times, instrument that first.

3. **Unowned knowledge.** Symptom: the agent confidently quotes an outdated policy. Cause: the retrieval corpus has no owner and no freshness process. Fix: every source the agent reads gets a named owner and an update SLA; stale documents are flagged in retrieval results (14. Knowledge Retrieval, 29. Advanced Production RAG).

4. **Over-broad credentials.** Symptom: a security review finds the agent's Stripe key can do anything. Cause: the pilot used an admin key "temporarily." Fix: a service identity with restricted scopes per tool, per-user authorization checks in the tool layer, and an audit log of every call (39. Agent Identity, Authorization, and Governance). Keep credentials out of the model's context.

5. **Sign-off on a small sample.** Symptom: production quality is visibly worse than the pilot report. Cause: the pilot stopped when the point estimate crossed the target. Fix: pre-register the sample size and sign off on the confidence interval's lower bound; include a slice analysis by ticket type, because averages hide a failing segment.

6. **No on-call owner.** Symptom: the agent breaks at 2 a.m. and the only person who understands it is the FDE, on another continent. Cause: handoff was a slide, not a rotation. Fix: the customer's team runs a real change end to end before GA; the runbook covers the kill switch, rollback of prompt and model versions, and how to read a trace; the pager points at them.

7. **Cost surprise.** Symptom: finance escalates in month two. Cause: cost was measured on the happy path at pilot volume. Fix: model cost per ticket including retries, long-tail tickets, and evaluation traffic; set budgets and alerts; use prompt caching and routing to cheaper models for easy tickets (16. Resource-Aware Optimization, 30. Fine-Tuning, Routing, and Prompt Optimization).

8. **Scope creep without re-gating.** Symptom: "while you're here, can it also do contract changes?" and the new capability ships without evals. Cause: additions bypassed the pipeline. Fix: every new workflow re-enters at scope and passes the same gates. The checklist is per workflow, not per customer.

## What changed by 2026

- **The role spread.** Forward-deployed teams moved from data-platform companies to AI labs and agent startups, because the gap between a capable model and a deployed agent is mostly integration, permissions, evals, and change management.
- **Connectors became protocols.** MCP servers for common enterprise systems replaced a lot of bespoke integration code. The MCP spec's current revision hardens authorization, and its official Enterprise Managed Authorization extension targets exactly the "security review for new API keys" problem. The FDE still has to scope what each server exposes.
- **Evals became the contract.** Customer-labeled golden sets, pre-agreed thresholds, and regression suites that run on every prompt or model change are now a normal part of a deployment agreement. Tracing platforms (LangSmith and others) made shadow-mode comparison and slice analysis routine.
- **Human oversight became a product surface.** Approval gates built on durable interrupts, approval UIs, and audit logs are expected in any agent that takes consequential actions. `create_agent`'s human-in-the-loop middleware and Deep Agents' `interrupt_on` package the pattern.
- **Agent identity became a governance topic.** Agents run as service identities with scoped, auditable permissions, and MCP's published roadmap lists agent identity as a focus area. Customers' security teams now ask about it in the first meeting.
- **Harnesses and managed runtimes reduced the build, not the deployment.** Hosted agent runtimes and harnesses handle loops, compaction, and sandboxes. Discovery, data access, sign-off, on-call, and cost still sit with the FDE.

## Interview questions

**Q1. A customer says, "We want an AI agent for support that issues refunds." Walk me through your first two weeks.**

Strong: I would not start building the refund agent. In week one I interview the decision owner, frontline agents, and security, map the ticket types and volumes, and get a baseline for handling time and reopen rate. I also start the data-access and security review immediately, because it is the longest pole. By the end of week two I want a signed one-page scope for one workflow, like invoice-dispute triage where the agent proposes refunds and a human approves them, with a metric, a target, a decision owner, and kill criteria. Full autonomy comes later, justified by approval data.

Weak: "I'd build a prototype with their data and show it to the VP."

**Q2. Design a customer-support agent that can take real actions, with escalation to humans.** *(reported at Sierra)*

Strong: Scope actions by risk. Reads are autonomous, reversible low-value writes are autonomous with audit, and money movement or account changes go through an approval gate implemented as a durable interrupt with the approver's identity recorded. The agent runs as a service identity with least-privilege scopes, and every tool call is logged with user, arguments, and result. Escalation triggers are explicit: low confidence, policy not found, customer asks for a human, or a sensitive intent. Escalations hand the human a summary and the trace. It rolls out through shadow mode, canary, and GA with evals at each gate.

Weak: "The agent escalates when it isn't sure."

**Q3. The pilot shows 86 percent agreement with humans on 50 tickets. The target was 85. Do you ship?**

Strong: Not yet. At n = 50 the 95 percent Wilson interval is roughly 74 to 93 percent, so the data is consistent with missing the target by a lot. I would extend shadow mode to the sample size we agreed in the scope, look at the disagreements by ticket type, and check outcome metrics like reopen rate. Some disagreements may be the agent being right, which I would have SMEs adjudicate. I would rather tell the owner "two more weeks" now than explain a production regression later.

Weak: "86 is above 85, so yes."

**Q4. How do you build an eval set when there is no labeled ground truth and experts are expensive?** *(reported at Cohere, Harvey)*

Strong: Start from historical tickets that humans already resolved, since the resolution is a label, and sample them stratified by ticket type and difficulty. Use expert time only where it is decisive: adjudicating disagreements between the agent and the historical resolution, and writing a rubric the rest of the labeling can follow. Use shadow mode to collect more comparisons for free. Calibrate any LLM grader against a small expert-labeled subset before trusting it. Grow the set from production failures.

Weak: "Use an LLM to generate test cases."

**Q5. How do you make an agent's actions auditable?** *(reported at Palantir)*

Strong: Every tool call is logged as a structured event with the service identity, the end user it acted for, arguments, result, timestamp, and the trace id of the run. Approvals record who approved, what they saw, and when. The graph's checkpoint history shows each decision point. Logs go to the customer's SIEM with their retention policy. Side-effecting tools use idempotency keys, so retries do not create duplicate actions. And the model's written reasoning is not treated as the audit record; the actions and their inputs are.

Weak: "We log the conversation."

**Q6. What is on your deployment-readiness checklist, and what makes an item "done"?**

Strong: Scope signed with a measured baseline; data agreements and corpus ownership; least-privilege service identity, per-user authorization, and audit logging; a customer-labeled golden set with thresholds met on a confidence interval and signed; approval gates and a tested kill switch; on-call, runbook, and traces owned by the customer; a cost model with alerts; and the customer's engineers having shipped a change themselves. An item is done only when there is evidence, like a link or a signed doc, and blocking items cannot be averaged away by a high overall score.

Weak: "Security review and testing."

**Q7. Your agent's scores improved on the eval set but the customer says it got worse. Why?** *(reported at Cognition)*

Strong: Several possibilities. The eval set drifted from production: new ticket types, a product launch, or a policy change the set does not cover. The metric is a proxy that the change optimized at the expense of what users feel, like shorter handling time but more reopens. Averages hide a segment that regressed. The grader may be biased toward the new outputs' style. Or latency and cost changed, which the eval does not measure. I would slice by ticket type, compare outcome metrics, refresh the set from recent traffic, and recalibrate the grader.

Weak: "The customer is probably wrong; the numbers went up."

## Build it yourself

Turn the pieces into an engagement tracker.

1. Add a `stage` argument to `readiness` and a `required_for` field on `Item` (`"pilot"`, `"canary"`, or `"production"`), so an item only counts toward stages at or after the one it is required for.
2. Add a `cost_model(tickets_per_month, calls_per_ticket, tokens_per_call, usd_per_mtok, cache_hit_rate)` function and a checklist item that is only done when modeled monthly cost is under a budget you pass in.
3. Extend `sign_off` with a per-slice check: split the shadow log by a `ticket_type` field and fail sign-off if any slice with at least 30 tickets has an agreement lower bound under the target minus 0.05.
4. Add a `reject` path test to the rollout graph: resuming with `{"decision": "reject", ...}` ends in stage `"rescope"` and the audit log records the rejecter.

Acceptance checks:

- `readiness(CHECKLIST, stage="pilot")` ignores production-only items and can return `READY` while the production verdict is `NOT READY`.
- `cost_model` with `cache_hit_rate=0.0` is strictly more expensive than with `0.5`, all else equal.
- A shadow log with one bad slice fails sign-off even when overall agreement passes.
- Resuming with a rejection gives `done["stage"] == "rescope"` and `rollout.get_state(cfg).next == ()`.

## Key takeaways

- The FDE's product is a deployed, owned, measured workflow, not a demo.
- Pick the pilot that clears the gates, not the one with the biggest headline value; stalled pilots deliver nothing.
- Start data access and security review in week zero; they are the longest pole.
- A checklist item is done only with evidence, and blocking items cannot be averaged away.
- Sign off on the confidence interval's lower bound at a pre-agreed sample size, with slices, not on a point estimate.
- Rollouts are long-running, human-gated processes; checkpoints and interrupts model them directly.
- Hand off on-call, runbooks, cost budgets, and the eval suite to the customer's team before you leave.
