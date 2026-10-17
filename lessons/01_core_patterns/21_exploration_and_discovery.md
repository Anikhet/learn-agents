# 21. Exploration and Discovery

*Track A (agent engineering). Book: Chapter 21 (Exploration and Discovery). Prerequisites: 00. What an agent actually is, 06. Planning, 17. Reasoning Techniques.*

Most of the patterns in this course assume the agent knows what it is looking for. A routing agent knows the categories. A planning agent knows the goal and decomposes it. Exploration is the pattern for when the agent does not know, and has to find out: which of five hypotheses explains the outage, which retrieval strategy works for this corpus, which of thirty candidate documents holds the answer. The agent must generate possibilities, test them cheaply, discard most of them, and stop when it has learned enough or spent enough.

That last clause is the whole difficulty. An exploring agent has no natural stopping point. Left alone it will keep finding interesting things. This lesson builds exploration as a bounded, logged search: hypothesis loops, the exploration-versus-exploitation trade-off, branch-and-evaluate over LangGraph's `Send`, and the exploration log that makes a research agent's output trustworthy.

## The problem

Wednesday. Twenty support tickets in six hours, all variants of:

> "Dashboard exports are timing out since this morning."

Nobody changed the export code. The ops agent has SQL, the deploy log, Stripe, Slack, and the observability API. It does not know the cause. A planning agent would try to write a plan, but every plan step depends on facts it does not have yet.

The naive approach is to let the model investigate freely:

```text
step 1   query the export job table ........................ 12 KB result
step 2   read the deploy log ................................ 30 KB result
step 3   search Slack for "export" .......................... 45 KB result
step 4   notice an unrelated Stripe webhook error, chase it .. 20 KB
step 5   search Slack for "webhook" ......................... 45 KB
...
step 23  context is 180 KB of raw tool output, the model has
         forgotten step 1, and it proposes redeploying the export
         service "to see if it helps"
```

Three failures, and they are distinct:

```text
1. No hypotheses.      Investigation without a candidate explanation is browsing.
2. No pruning.         Every lead was followed to the end; nothing was discarded early.
3. No budget or log.   The run ended when the context filled, and no record survives
                       of what was ruled out or why.
```

Exploration done well looks like a scientist's notebook, not a browser history:

```text
hypotheses   H1 database slow      H2 new deploy      H3 one huge tenant     H4 upstream API
cheap test   p95 query latency     diff deploy log    exports by tenant      status page
result       normal (RULED OUT)    no deploy 48h      one tenant = 71%       green
             (RULED OUT)           of rows (SUPPORTED) (RULED OUT)
refine       H3a tenant added 4M rows overnight   H3b tenant's export filter regressed
cheap test   row count delta       filter version
result       +4.1M rows (CONFIRMED)
stop         confidence 0.9 >= threshold, budget 6/10 tests used
```

The book frames exploration as an agent searching a solution space with bounded experiments and a log of what was tried. The systems view: this is search with a budget, and the interesting design questions are the ones search always has. How do you generate candidates? How do you evaluate them cheaply? How wide do you branch? When do you stop exploring and commit? And how do you make the search auditable?

### Exploration versus exploitation

Every exploring agent faces the same trade-off at every step: spend the next test on the leading hypothesis (exploit) or on an untested one (explore). Always exploiting means the first plausible story wins and never gets challenged. Always exploring means nothing gets confirmed.

The classic tool is the multi-armed bandit. Each hypothesis (or strategy) is an arm; each test is a pull with a noisy reward. Upper Confidence Bound (UCB) picks the arm with the highest `mean + c * sqrt(ln(total_pulls) / pulls_for_this_arm)`: arms that have been tried rarely get a bonus, so nothing is abandoned on one bad result, but the bonus shrinks as evidence accumulates. Agents use the same idea when choosing which retrieval strategy to run, which subagent to route to, or which hypothesis to test next.

```text
model:        "Three explanations fit. H3 looks strongest."
application:  UCB(H1)=0.4, UCB(H2)=0.7, UCB(H3)=0.85  -> test H3; log the scores
application:  test result 0.9 -> update H3; budget 7/10 remaining
model:        "H3 confirmed by row-count delta. Refining into H3a, H3b."
```

The model proposes and interprets. The code selects, tests, updates, and stops.

## Build it without a framework

The scripted model helper, copied verbatim, is used in the LangGraph section where the model proposes hypotheses. The exploration machinery itself is deterministic code.

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


print("scripted model ready")
```

### The hypothesis loop: generate, test, prune, refine, stop

The loop below is the notebook from the diagram. Each hypothesis has a cheap test (a tool call in production; a lookup table here) that returns a support score in [0, 1]. Hypotheses below a floor are pruned. A supported hypothesis can be refined into children. The loop stops on confidence, on budget, or when nothing remains. Every action is appended to a log.

```{code-cell} python
# What the tools would return for each hypothesis's cheap test. Support in [0, 1].
EVIDENCE = {
    "H1 database is slow":            (0.10, "p95 query latency 42ms, normal"),
    "H2 recent deploy regressed":     (0.05, "no deploy in 48h"),
    "H3 one tenant dominates load":   (0.70, "tenant 8812 = 71% of export rows today"),
    "H4 upstream API degraded":       (0.10, "status page green, latency normal"),
    "H3a tenant bulk-imported rows":  (0.92, "tenant 8812 row count +4.1M since 02:00"),
    "H3b tenant export filter broke": (0.20, "filter version unchanged"),
}
REFINEMENTS = {"H3 one tenant dominates load": ["H3a tenant bulk-imported rows",
                                                 "H3b tenant export filter broke"]}


def cheap_test(hypothesis):
    """Stand-in for a tool call. Returns (support, evidence)."""
    return EVIDENCE[hypothesis]


def explore(hypotheses, budget=10, prune_below=0.3, confident_at=0.9):
    log, open_set, results, tests = [], list(hypotheses), {}, 0
    while open_set and tests < budget:
        h = open_set.pop(0)
        support, evidence = cheap_test(h)
        tests += 1
        results[h] = support
        if support < prune_below:
            log.append({"test": tests, "hypothesis": h, "support": support, "action": "pruned", "evidence": evidence})
            continue
        if support >= confident_at:
            log.append({"test": tests, "hypothesis": h, "support": support, "action": "confirmed", "evidence": evidence})
            return {"conclusion": h, "confidence": support, "tests": tests, "log": log, "stopped": "confidence"}
        children = REFINEMENTS.get(h, [])
        log.append({"test": tests, "hypothesis": h, "support": support,
                    "action": f"supported; refined into {len(children)}", "evidence": evidence})
        # Refinements go to the front: exploit the promising branch before testing the rest.
        open_set = children + open_set
    best = max(results, key=results.get) if results else None
    return {"conclusion": best, "confidence": results.get(best, 0.0), "tests": tests, "log": log,
            "stopped": "budget" if tests >= budget else "exhausted"}


outcome = explore(["H1 database is slow", "H2 recent deploy regressed",
                   "H3 one tenant dominates load", "H4 upstream API degraded"])
for row in outcome["log"]:
    print(f"#{row['test']} {row['hypothesis']:34s} {row['support']:.2f} {row['action']:28s} | {row['evidence']}")
print(f"\nconclusion: {outcome['conclusion']} (confidence {outcome['confidence']}, "
      f"{outcome['tests']} tests, stopped on {outcome['stopped']})")
assert outcome["conclusion"] == "H3a tenant bulk-imported rows"
assert outcome["stopped"] == "confidence" and outcome["tests"] == 4
assert "H4 upstream API degraded" not in {r["hypothesis"] for r in outcome["log"]}   # never needed
assert [r["action"] for r in outcome["log"]][:2] == ["pruned", "pruned"]
```

Four tests, not six. H4 was never run because H3's refinement was pushed to the front of the queue and its child crossed the confidence threshold. That is exploitation working correctly: once a branch is promising, dig there before testing unrelated alternatives. The log still records that H1 and H2 were ruled out, with the evidence, which is exactly what the incident report needs.

Change `confident_at` to 0.95 and the loop continues to H3b and then H4, exhausts the open set, and reports the best-supported hypothesis with `stopped == "exhausted"`. Change `budget` to 2 and it stops after pruning H1 and H2 with a low-confidence conclusion. Both outcomes are legitimate results of an exploration; what would be illegitimate is reporting them with the same confidence as the confirmed case.

### Choosing what to test next: a bandit over strategies

The loop above used a fixed ordering plus "refinements first." When tests are noisy, that is fragile: one unlucky result prunes the right hypothesis. The bandit framing handles noise by keeping a running mean per arm and an exploration bonus that shrinks with evidence.

Here the arms are retrieval strategies for the agent's knowledge base (the same problem 14. Knowledge Retrieval faces): keyword search, dense search, and hybrid. Each pull runs one query and gets a noisy relevance score. Watch how UCB spends early pulls across all arms and later concentrates on the best one, without ever fully abandoning the others.

```{code-cell} python
import math
import random

TRUE_QUALITY = {"bm25": 0.55, "dense": 0.62, "hybrid": 0.78}   # unknown to the agent


def pull(arm, rng):
    """One noisy evaluation of a strategy: a relevance score in [0, 1]."""
    return min(1.0, max(0.0, rng.gauss(TRUE_QUALITY[arm], 0.15)))


def ucb_explore(arms, pulls_budget=100, c=0.5, seed=3):
    rng = random.Random(seed)
    counts = {a: 0 for a in arms}
    means = {a: 0.0 for a in arms}
    history = []
    for t in range(1, pulls_budget + 1):
        untried = [a for a in arms if counts[a] == 0]
        if untried:
            arm = untried[0]                                  # every arm gets one pull first
        else:
            arm = max(arms, key=lambda a: means[a] + c * math.sqrt(math.log(t) / counts[a]))
        reward = pull(arm, rng)
        counts[arm] += 1
        means[arm] += (reward - means[arm]) / counts[arm]     # running mean
        history.append(arm)
    return counts, means, history


counts, means, history = ucb_explore(["bm25", "dense", "hybrid"])
print("pulls per arm:", counts)
print("estimated quality:", {a: round(m, 2) for a, m in means.items()})
print("first 12 choices:", history[:12])
print("last 12 choices: ", history[-12:])
best = max(means, key=means.get)
assert best == "hybrid"
assert counts["hybrid"] > counts["bm25"] and counts["hybrid"] > counts["dense"]
assert all(counts[a] >= 3 for a in counts)          # nothing was abandoned on one bad pull
assert history[-12:].count("hybrid") >= 8           # exploitation dominates late
```

The first twelve choices bounce between arms; the last twelve are mostly `hybrid`. That shift from exploration to exploitation is not scheduled; it emerges from the confidence bonus shrinking as `counts[a]` grows. For an agent, the arms might be hypotheses, subagents, tools, or prompts, and the reward is whatever cheap evaluation you can run. The pattern is the same: never decide on one sample, but let evidence concentrate effort.

## How LangGraph does it

The from-scratch loop tested one hypothesis at a time. When the cheap tests are independent tool calls, run them in parallel: branch out to one test per hypothesis, collect the results, prune, and either refine or stop. In LangGraph that is `Send` for the fan-out, a reducer for the fan-in, and a conditional edge for the loop decision. The exploration log lives in state with an append reducer, which is what makes it survive across rounds and (with a checkpointer) across restarts.

```text
START -> propose --Send x N--> test (parallel, one per hypothesis)
                                   |
                                   v
                                evaluate --(confident or budget spent)--> report -> END
                                   |
                                   +--(promising branch, budget left)--> propose (refine)
```

The `propose` node is where the model contributes: given the task and the log so far, it proposes hypotheses or refinements. The scripted model plays that role with two turns: an initial set and one refinement. Everything else is code.

```{code-cell} python
import json
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class ExploreState(TypedDict):
    task: str
    frontier: list[str]                              # hypotheses to test this round (replaced)
    results: Annotated[dict, lambda a, b: {**a, **b}]  # hypothesis -> support (merged)
    log: Annotated[list[dict], operator.add]         # append-only exploration log
    tests: Annotated[int, operator.add]              # budget consumed
    conclusion: str | None


BUDGET, PRUNE_BELOW, CONFIDENT_AT = 8, 0.3, 0.9

# Turn 1: initial hypotheses. Turn 2: refinements of the supported one. Both as JSON lists.
proposer = scripted(
    json.dumps(["H1 database is slow", "H2 recent deploy regressed",
                "H3 one tenant dominates load", "H4 upstream API degraded"]),
    json.dumps(["H3a tenant bulk-imported rows", "H3b tenant export filter broke"]),
)


def propose(state: ExploreState):
    """The model proposes; the log so far is its input so it does not repeat pruned ideas."""
    ruled_out = [h for h, s in state["results"].items() if s < PRUNE_BELOW]
    prompt = f"Task: {state['task']}\nRuled out: {ruled_out}\nPropose hypotheses as a JSON list."
    hypotheses = json.loads(proposer.invoke([HumanMessage(prompt)]).content)
    return {"frontier": hypotheses,
            "log": [{"round": "propose", "hypotheses": hypotheses}]}


def fan_out(state: ExploreState):
    """One Send per hypothesis: each becomes a parallel `test` invocation with its own input."""
    return [Send("test", {"hypothesis": h}) for h in state["frontier"]]


def test(payload: dict):
    """Runs in parallel. The payload is the Send input, not the graph state."""
    support, evidence = cheap_test(payload["hypothesis"])
    return {"results": {payload["hypothesis"]: support}, "tests": 1,
            "log": [{"round": "test", "hypothesis": payload["hypothesis"],
                     "support": support, "evidence": evidence}]}


def evaluate(state: ExploreState):
    best = max(state["frontier"], key=lambda h: state["results"][h])
    if state["results"][best] >= CONFIDENT_AT:
        return {"conclusion": best, "log": [{"round": "evaluate", "decision": "confirmed", "best": best}]}
    return {"log": [{"round": "evaluate", "decision": "refine" if state["results"][best] >= PRUNE_BELOW
                     else "exhausted", "best": best, "budget_left": BUDGET - state["tests"]}]}


def next_step(state: ExploreState):
    if state["conclusion"] or state["tests"] >= BUDGET:
        return "report"
    best = max(state["frontier"], key=lambda h: state["results"][h])
    return "propose" if state["results"][best] >= PRUNE_BELOW else "report"


def report(state: ExploreState):
    if state["conclusion"]:
        return {}
    best = max(state["results"], key=state["results"].get)     # best effort, flagged as such
    return {"conclusion": f"UNCONFIRMED: {best}",
            "log": [{"round": "report", "decision": "best effort", "best": best}]}


builder = StateGraph(ExploreState)
builder.add_node("propose", propose)
builder.add_node("test", test)
builder.add_node("evaluate", evaluate)
builder.add_node("report", report)
builder.add_edge(START, "propose")
builder.add_conditional_edges("propose", fan_out, ["test"])
builder.add_edge("test", "evaluate")                # runs once, after all parallel tests land
builder.add_conditional_edges("evaluate", next_step, ["propose", "report"])
builder.add_edge("report", END)
graph = builder.compile()

out = graph.invoke({"task": "Dashboard exports timing out since this morning",
                    "frontier": [], "results": {}, "log": [], "tests": 0, "conclusion": None})
for row in out["log"]:
    print(row)
print(f"\nconclusion: {out['conclusion']}  tests: {out['tests']}/{BUDGET}")
assert out["conclusion"] == "H3a tenant bulk-imported rows"
assert out["tests"] == 6                                        # 4 parallel, then 2 parallel
assert sum(1 for r in out["log"] if r["round"] == "propose") == 2
assert all(r["support"] < PRUNE_BELOW for r in out["log"]
           if r["round"] == "test" and r["hypothesis"] in ("H1 database is slow", "H2 recent deploy regressed"))
```

Map the primitives back to the from-scratch loop:

| From scratch | LangGraph | Underneath |
|---|---|---|
| `open_set.pop(0)`; one test per iteration | `fan_out` returning `Send("test", ...)` per hypothesis | N parallel `test` runs in one super-step, each with its own input dict |
| `results[h] = support` | `results` key with a dict-merge reducer | parallel branches write disjoint keys; the reducer merges them |
| `log.append(...)` | `log` with `operator.add` | every branch appends; order within a super-step is not guaranteed, so rows carry their own labels |
| `tests += 1` | `tests` with `operator.add` | each parallel branch adds 1; the sum is the budget consumed |
| `while open_set and tests < budget` | `next_step` conditional edge | routing back to `propose` is the loop; `report` is termination |
| refinements pushed to the front | `propose` called again with the log | the model sees what was ruled out and refines the survivor |

Two details deserve attention. `test` receives the `Send` payload, not the graph state, which is what lets it run for a hypothesis that is not a state key. And `evaluate` runs once after all `test` branches complete, because LangGraph waits for every task in a super-step before starting the next node. That is the fan-in; there is no explicit join node.

Notice the cost difference. The sequential loop used 4 tests to reach the same conclusion; the parallel version used 6, because it tested H4 and H3b that the sequential version skipped. Parallel exploration trades tokens for wall-clock time. When tests are cheap and latency matters (an incident), fan out. When tests are expensive (a subagent doing a twenty-minute investigation each), the sequential loop with a bandit is the better shape.

### Research agents: exploration over documents

The same graph runs a research agent. Hypotheses become search queries or candidate sources; `test` becomes "fetch and extract the relevant claim with a citation"; `evaluate` becomes "do the extracted claims answer the question, and do they agree?" The exploration log becomes the bibliography and the list of sources examined and rejected. Deep Agents and similar harnesses implement this with subagents: each branch runs in its own context, reads the whole document, and returns a condensed finding so the parent context sees a paragraph, not the document. 34. Context Engineering treats that isolation directly; 26. Map-Reduce Summarization treats the aggregation.

```{code-cell} python live
import os
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model


class Hypotheses(BaseModel):
    hypotheses: list[str] = Field(description="3 to 5 distinct, cheaply testable explanations")
    cheapest_test_for_each: list[str] = Field(description="one tool call or query per hypothesis")


real_model = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
proposer_live = real_model.with_structured_output(Hypotheses)

ruled_out = ["H1 database is slow (p95 42ms)", "H2 recent deploy (none in 48h)"]
proposal = proposer_live.invoke(
    "Incident: dashboard exports timing out since 06:00 for ~20 tenants. Tools: SQL over export jobs "
    "and tenant tables, deploy log, Stripe, Slack search, observability API.\n"
    f"Already ruled out: {ruled_out}\n"
    "Propose distinct hypotheses, each with the single cheapest test that would rule it in or out.")
for h, t in zip(proposal.hypotheses, proposal.cheapest_test_for_each):
    print(f"- {h}\n    test: {t}")
```

The live prompt asks for hypotheses *with* their cheapest test, and it passes the ruled-out list. Those two moves are what turn a model's brainstorm into search input: the tests are what `Send` fans out over, and the ruled-out list stops the proposer from re-proposing a pruned branch.

## What goes wrong in production

1. **Unbounded exploration.** Symptom: a research run costs $60 and returns a document that mentions everything and concludes nothing. Cause: the only stop condition was "the model decides it is done," and a model exploring an open space rarely does. Fix: budgets in code (tests, tokens, wall-clock) plus a confidence threshold, and a `report` node that produces a best-effort conclusion flagged as unconfirmed when the budget ends first.

2. **Premature commitment.** Symptom: the first plausible hypothesis becomes the conclusion; the post-incident review finds the real cause was on the untested list. Cause: pure exploitation, often because the model's first proposal was fluent. Fix: test the initial frontier before refining, use a bandit-style bonus for untested branches when tests are noisy, and require that the confirmed hypothesis's support come from evidence, not from the model's stated confidence (17. Reasoning Techniques).

3. **Novelty chasing.** Symptom: the agent finds an interesting unrelated bug and spends the budget on it. Cause: the evaluator scores "interesting" instead of "explains the task." Fix: every hypothesis is scored against the task statement; the `propose` prompt includes the task and the ruled-out list every round; anything discovered off-task is enqueued as a separate task (20. Prioritization) rather than pursued now.

4. **Lost exploration log.** Symptom: three days later someone asks "did we check the upstream API?" and nobody can tell. Cause: the log lived in the transcript and was compacted away. Fix: the log is a typed state key with an append reducer, persisted by the checkpointer, and rendered into the final report as "examined and ruled out, with evidence."

5. **Parallel branches with side effects.** Symptom: four hypothesis tests each restart the export service "to see." Cause: fan-out over tests that are not read-only. Fix: exploration tests must be observations. Anything that mutates goes through the approval gate and never runs in a `Send` fan-out. Tag tools as read-only versus side-effecting and assert it in `fan_out`.

6. **Tool results flood the context.** Symptom: quality collapses at round three because each test returned 30 KB. Cause: raw tool output appended to the transcript. Fix: `test` returns a support score and a one-line evidence string; the raw payload is written to a store or file with an identifier the model can re-fetch. This is the tool-result offloading pattern of 34. Context Engineering.

7. **Non-reproducible runs.** Symptom: the same incident explored twice yields different conclusions. Cause: sampling in the proposer plus noisy tests plus no seed. Fix: log the proposer's outputs and the test results so a run can be replayed; evaluate the exploration agent on a fixed incident set with pass rates over multiple runs (22. Evaluation Reliability), and treat disagreement between runs as a signal to raise the confidence threshold.

## What changed by 2026

- **Exploration is the case for agents.** Anthropic's guidance on building agents draws the line at open-ended problems where the path cannot be predetermined; that is a definition of exploration. Everything with a known path should be a workflow. Exploration is where you accept the cost and unpredictability of an agent because there is no alternative, which is also why it needs the tightest budgets.

- **Subagents made branches context-isolated.** Harnesses now run each exploratory branch in its own context and return a condensed finding. Anthropic's context-engineering guidance describes sub-agents returning summaries of roughly one to two thousand tokens; Deep Agents exposes a `task` tool that spawns subagents, with parallel calls supported. The `Send` fan-out you built is the runtime mechanism underneath; the subagent is a branch whose transcript is thrown away and whose finding is kept.

- **Reasoning models explore internally, so the external loop got coarser.** A reasoning model will consider several hypotheses inside one call. The external exploration loop is now about what a single call cannot do: run tools in parallel, persist a log, enforce a budget, and isolate context. Do not script the model's internal thinking; script the tests and the stopping rule.

- **Evaluation of research agents moved to trajectories.** It is not enough to grade the final report. Current eval practice, including Anthropic's writing on agent evals, grades the trajectory: were the right sources examined, were pruned branches pruned for the right reason, was the budget respected. The exploration log is the artifact those evals read (37. Trajectory Evals).

- **Tool discovery became exploration too.** With MCP servers exposing hundreds of tools and the current spec's cacheable tool listings, harnesses now search for tools instead of loading them all; LangChain ships tool-selection middleware for this. Which tool to try is itself an explore-versus-exploit decision, made with the same machinery.

## Interview questions

**Q1. How do you keep an exploring agent from running forever?**

Strong: Stopping is enforced in code with three independent conditions: a confidence threshold on the leading hypothesis, a hard budget on tests, tokens, and wall-clock, and exhaustion of the frontier. When the budget ends first, the agent still produces a report, but the conclusion is explicitly flagged as unconfirmed with the best-supported hypothesis and the evidence gathered. The model never decides when to stop; it proposes and interprets. I also make the confidence come from evidence scores produced by tests, not from the model's self-reported certainty, because reasoning models' stated confidence is weak evidence.

Weak: "Tell the model to stop when it has found the answer."

**Q2. Explain exploration versus exploitation in the context of an agent, with a concrete mechanism.**

Strong: At every step an agent choosing what to test next can dig into the leading hypothesis or try an untested one. Pure exploitation locks in the first plausible story; pure exploration never confirms anything. A concrete mechanism is UCB: score each option by its running mean plus a bonus proportional to `sqrt(ln(total) / count)`, so rarely tried options get a boost that shrinks as evidence accumulates. Early pulls spread across options, late pulls concentrate on the best. In agents the options might be hypotheses, retrieval strategies, tools, or subagents, and the reward is whatever cheap evaluation exists. The key property is that nothing is abandoned on one noisy result.

Weak: "Use temperature to make the model more creative."

**Q3. Design a root-cause investigation agent for production incidents.**

Strong: It is a bounded hypothesis loop. The model proposes three to five distinct hypotheses each with its cheapest read-only test, given the incident and the list already ruled out. Code fans the tests out in parallel, each returning a support score and one line of evidence, with raw payloads offloaded to storage. An evaluator prunes below a floor, confirms above a threshold, or asks the model to refine the survivor. Budgets on tests and time are enforced in code. Everything goes into an append-only log in persisted state: proposals, results, prunes, decisions. The output is the confirmed cause with evidence, plus the ruled-out list, which the incident review needs as much as the answer. Any remediation is a separate task behind an approval gate, never a test.

Weak: "Give the model access to the logs and ask it what went wrong."

**Q4. Why should exploratory tests be read-only, and how do you enforce that?**

Strong: Exploration fans out and repeats; a side effect inside a test runs several times, in parallel, and possibly on a hypothesis that turns out to be wrong. Restarting a service "to see" during four parallel tests is an incident inside the incident. I enforce it at the tool layer: tools carry a read-only or side-effecting tag, the fan-out asserts every scheduled test uses read-only tools, and side-effecting tools require an approval interrupt so they cannot run inside a parallel branch at all. The model can propose a mutation as a hypothesis, but it becomes a task in the queue, not a test.

Weak: "Prompt the model not to change anything."

**Q5. When do you parallelize hypothesis tests versus run them sequentially?**

Strong: Parallel when tests are cheap and independent and latency matters, as in an incident where four SQL queries can run at once. Sequential when tests are expensive, such as a subagent investigation that takes minutes and thousands of tokens, because the sequential loop can exploit early findings and skip tests that a promising branch makes unnecessary. In my example the sequential loop reached the answer in four tests and the parallel one in six; parallel bought wall-clock time with tokens. The decision is a budget question. In LangGraph both shapes use the same state; the difference is whether the frontier goes through a `Send` fan-out or is popped one at a time.

Weak: "Always parallelize; it is faster."

**Q6. How do you evaluate a research or exploration agent?**

Strong: Grade the trajectory, not only the conclusion. From the exploration log I check: were the hypotheses distinct and relevant to the task, was each pruned branch pruned on evidence, were the right sources examined, was the budget respected, was the conclusion's confidence justified by the support scores. Then the outcome: on a fixed set of incidents or research questions with known answers, the pass rate over several runs, since runs are non-deterministic. I also track cost per confirmed conclusion and the unconfirmed rate. If two runs on the same input reach different conclusions, that is a calibration signal to raise the confidence threshold or improve the tests.

Weak: "Check whether the final answer is correct."

**Q7. What goes in an exploration log and why does it need to be state rather than transcript?**

Strong: Each proposal round with its hypotheses, each test with the hypothesis, support score, and one line of evidence, each evaluation decision (prune, refine, confirm, budget exhausted), and the running budget. It needs to be a typed state key with an append reducer for three reasons: parallel branches must be able to append without overwriting each other, the transcript gets compacted and the log must not, and the final report and the trajectory evals read it as structured data. With a checkpointer it also survives crashes, so a twenty-minute investigation that dies at minute nineteen resumes with its ruled-out list intact.

Weak: "The model remembers what it tried."

## Build it yourself

Extend the LangGraph exploration graph into a budget-aware, offloading research loop.

1. Add a `raw` store: `test` writes the full evidence payload to a dict keyed by hypothesis (stand-in for a file or Store) and returns only the one-line summary to state. Add a `fetch_evidence(hypothesis)` tool the model could use to re-read it.
2. Add per-round wall-clock accounting: each `test` adds a simulated `cost_ms` to a `elapsed_ms` key with `operator.add`; the `next_step` edge must also stop when `elapsed_ms` exceeds a limit.
3. Make `fan_out` refuse any hypothesis whose test is tagged as side-effecting (add a `SIDE_EFFECTING` set) and log the refusal.
4. Add a `budget-first` scenario: set `BUDGET = 4` so the loop ends before refinement, and make `report` produce an `UNCONFIRMED:` conclusion with the ruled-out list attached.

Acceptance checks:

- With the default budget the conclusion is `H3a tenant bulk-imported rows`, `tests == 6`, and the raw store has exactly six entries while no log row exceeds 120 characters.
- With `BUDGET = 4` the conclusion starts with `UNCONFIRMED:` and names `H3 one tenant dominates load`; the log contains exactly one `propose` round.
- A hypothesis whose test is in `SIDE_EFFECTING` never appears in a `test` log row and does appear in a `refused` row.
- Setting the wall-clock limit below the first round's total `cost_ms` ends the run at `report` after one round with `stopped_on == "time"` in the final log row.

## Key takeaways

- Exploration is search with a budget: generate hypotheses, test them cheaply, prune, refine the survivor, and stop on confidence, budget, or exhaustion.
- The model proposes and interprets; code selects the next test, updates support, enforces the budget, and decides when to stop.
- Balance exploration and exploitation deliberately; a bandit-style bonus for untested options prevents both premature commitment and endless browsing.
- Fan out with `Send` when tests are cheap, read-only, and latency matters; loop sequentially when tests are expensive and early findings can prune later work.
- Tests must be observations; anything that mutates becomes a task behind an approval gate, never a parallel branch.
- The exploration log is typed state with an append reducer: it survives compaction and crashes, feeds the report, and is what trajectory evals grade.
- An unconfirmed conclusion is a legitimate result; reporting it with confirmed-level confidence is the failure.
