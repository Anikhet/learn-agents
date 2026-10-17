# 04. Reflection

*Track A (agent engineering). Book: Chapter 4 (Reflection). Prerequisites: 00. What an agent actually is; 01. Prompt Chaining.*

Reflection is the pattern where an output is produced, examined, and revised before it is used. The book presents it as a generate-critique-refine loop and it is genuinely useful. It is also the pattern most often built wrong, because the obvious implementation, asking the model to check its own work, is the weakest form of it. This lesson builds the loop with nothing hidden, compares three kinds of verifier (the same model, a different model, and a tool that actually runs the thing), rebuilds the loop as a LangGraph evaluator-optimizer graph, and explains why "bounded" is not optional.

## The problem

The support agent has been asked to write a SQL query:

> "How many Acme seats were active in September?"

It drafts:

```sql
SELECT COUNT(*) FROM seats WHERE customer = 'Acme' AND active = true
```

The query runs. It returns 14. The agent reports 14. Two things are wrong and nothing caught either: there is no date filter (so this is *today's* active seats, not September's), and `COUNT(*)` counts rows, not distinct users, and the `seats` table has one row per seat-month. The answer is confidently wrong, and it was produced in one shot by a system that never looked back.

```text
draft ──► run ──► 14 ──► "Acme had 14 active seats in September."
                            ▲
                            │ nobody asked: does this query answer the question?
```

The fix the book proposes is a second pass. Something looks at the draft against the requirements and says what is wrong; the draft is revised; repeat until it passes or a budget runs out:

```text
                ┌──────────────────────────────────────┐
                ▼                                      │
requirements ──► generate ──► draft ──► critique ──► issues? ──yes──┘
                                            │
                                           no
                                            ▼
                                        accept
```

Now the design question, and it is the whole lesson: **who is the critic?**

| Critic | What it is | What it catches | What it misses |
|---|---|---|---|
| Self-reflection | same model, "review your answer" | style, missing requirements it was told about, obvious slips | anything it believed the first time; shares every blind spot with the generator |
| Cross-model / separate critic | a different model or a fresh context with a rubric | more of the above, some factual and logical errors | still a language model guessing whether SQL is right |
| Tool-based verifier | run the SQL, run the tests, check the schema, compare to ground truth | whether the output actually works | nothing about taste; needs an executable check to exist |

The rows go from cheap and weak to expensive and strong. The consistent production finding, and Anthropic's harness-design guidance says this plainly, is that generation and evaluation should be separated because a model grading its own work is unreliable. The strongest evaluators are external: a test suite, a compiler, a database, a rubric applied by a fresh context. Self-reflection is a polish step, not a correctness step.

Why is self-reflection weak? The model that wrote `COUNT(*)` without a date filter did so because, in that context, that looked right. Asking the same context "is this right?" asks the same beliefs the same question. It will sometimes catch a slip. It will rarely catch a misconception. And there is a second reason: a model's written self-critique is text it generated, not a faithful report of a checking process. Anthropic's research on reasoning faithfulness found models frequently do not verbalize what actually drove their answers. So "I have double-checked and this is correct" is a sentence, not evidence. **Verify the outcome, not the prose about the outcome.**

## Build it without a framework

The scripted model plays the generator and, in one variant, the critic. A tiny in-memory "database" plays the tool-based verifier.

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

### A verifier that runs the thing

The verifier is the most important component, so it comes first. This one does not read the SQL and form an opinion. It checks structural requirements that the question implies, then executes the query against a fixture and compares the result to a known answer. Every check returns a specific, actionable message, because the message is what goes back to the generator.

```{code-cell} python
import re
import sqlite3

db = sqlite3.connect(":memory:")
db.executescript("""
CREATE TABLE seats (customer TEXT, user_id TEXT, month TEXT, active INTEGER);
INSERT INTO seats VALUES
 ('Acme','u1','2026-09',1),('Acme','u2','2026-09',1),('Acme','u3','2026-09',0),
 ('Acme','u1','2026-10',1),('Acme','u2','2026-10',1),('Acme','u4','2026-10',1),
 ('Globex','u9','2026-09',1);
""")
EXPECTED = 2   # Acme, September, distinct active users


def verify_sql(query):
    """Tool-based critic: structural checks, then execute and compare. Returns a list of issues."""
    issues = []
    q = query.lower()
    if "month" not in q:
        issues.append("no date filter: the question asks about September; filter on month = '2026-09'")
    if "count(*)" in q:
        issues.append("count(*) counts seat-month rows; the question asks for users, use COUNT(DISTINCT user_id)")
    if issues:
        return issues                       # do not run a query we already know is wrong
    if not re.fullmatch(r"\s*select\b[^;]*;?\s*", q):
        return ["only a single SELECT statement is allowed"]
    try:
        (value,) = db.execute(query).fetchone()
    except sqlite3.Error as exc:
        return [f"execution error: {exc}"]
    if value != EXPECTED:
        issues.append(f"result {value} does not match the fixture's expected value {EXPECTED}")
    return issues


bad = "SELECT COUNT(*) FROM seats WHERE customer = 'Acme' AND active = 1"
good = "SELECT COUNT(DISTINCT user_id) FROM seats WHERE customer = 'Acme' AND month = '2026-09' AND active = 1"
print(verify_sql(bad))
print(verify_sql(good))
assert len(verify_sql(bad)) == 2
assert verify_sql(good) == []
```

Look at what the issues say. Not "this looks wrong" but "no date filter: filter on month = '2026-09'". A critique the generator can act on is a critique that names the requirement and the fix. Vague critiques ("could be improved") produce vague revisions and burn iterations.

### The bounded loop

Now the loop. Generate, verify, and if there are issues, regenerate with the issues attached. Three properties are non-negotiable and each is a line of code:

```text
max_rounds          the loop ends even if the critic is never satisfied
best-so-far         if the budget runs out, return the draft with the fewest issues, not the last one
history in prompt   the generator sees what was already tried, so it does not oscillate
```

```{code-cell} python
from langchain_core.messages import HumanMessage, SystemMessage


def reflect_loop(generator, verifier, task, max_rounds=3):
    """generate -> verify -> revise, bounded, returning the best draft and a trace."""
    prompt = [SystemMessage("Write one SQL query that answers the question. Return only SQL."),
              HumanMessage(task)]
    best, trace = None, []
    for round_no in range(1, max_rounds + 1):
        draft = generator.invoke(prompt).content.strip()
        issues = verifier(draft)
        trace.append({"round": round_no, "issues": len(issues)})
        if best is None or len(issues) < len(best[1]):
            best = (draft, issues)
        if not issues:
            return draft, True, trace
        # The critique goes back as a new turn; the failed draft stays visible so the model
        # does not resubmit it.
        prompt += [AIMessage(draft), HumanMessage("Revise. Issues:\n- " + "\n- ".join(issues))]
    return best[0], False, trace


generator = scripted(
    "SELECT COUNT(*) FROM seats WHERE customer = 'Acme' AND active = 1",
    "SELECT COUNT(*) FROM seats WHERE customer = 'Acme' AND month = '2026-09' AND active = 1",
    "SELECT COUNT(DISTINCT user_id) FROM seats WHERE customer = 'Acme' AND month = '2026-09' AND active = 1",
)
final, accepted, trace = reflect_loop(generator, verify_sql,
                                      "How many Acme users had an active seat in September 2026?")
for t in trace:
    print(t)
print(final)
assert accepted and [t["issues"] for t in trace] == [2, 1, 0]

# Budget exhaustion: a generator that never fixes the date filter. We get the best draft, not a crash.
stubborn = scripted(*["SELECT COUNT(*) FROM seats WHERE customer = 'Acme' AND active = 1"] * 2)
final2, accepted2, trace2 = reflect_loop(stubborn, verify_sql, "same question", max_rounds=2)
print("accepted:", accepted2, "rounds:", len(trace2))
assert not accepted2 and len(trace2) == 2
```

Round 1 had two issues, round 2 fixed one, round 3 fixed the other. Real loops look like this: each round addresses what it was told, and the verifier catches the next layer. That is why the critique must be complete each round (list every issue, not just the first) and why the loop needs more than one iteration to be worth building.

The stubborn case matters as much. The loop ended after two rounds with `accepted=False`. What the caller does with that is a product decision, escalate, ask a human, refuse to answer, but the loop's job was to stop and say so. An unbounded reflection loop with a critic that is never satisfied is the reflection pattern's version of the runaway agent from lesson 00.

### Three critics, same loop

The loop does not care who the critic is. Swap the verifier and you have the other two rows of the table. The self-critic asks the generating model to review; the cross-model critic asks a separate model with a rubric. Both return the same `list[str]` shape, so they compose with the tool verifier: run the cheap structural check first, then the executable check, then a model critic for the qualities no tool can check (clarity, tone, whether the explanation matches the query).

```{code-cell} python
import json


def model_critic(critic_model, rubric):
    """A separate model with an explicit rubric. Returns the same list[str] shape as verify_sql."""
    def verify(draft):
        raw = critic_model.invoke(
            f"Rubric: {rubric}\nReview the draft against each rubric item. "
            f"Return JSON: {{\"issues\": [..]}} with one actionable item per failure.\n\n{draft}").content
        return json.loads(raw)["issues"]
    return verify


def combined(*verifiers):
    """Cheap and decisive checks first; stop at the first layer that finds issues."""
    def verify(draft):
        for v in verifiers:
            issues = v(draft)
            if issues:
                return issues
        return []
    return verify


rubric = "uses explicit column list or a single aggregate; no SELECT *; readable aliases"
critic = model_critic(scripted('{"issues": []}'), rubric)
verifier = combined(verify_sql, critic)          # tool first, then taste

issues = verifier(good)
print("combined verdict on good query:", issues)
assert issues == []

# The tool layer short-circuits: a wrong query never reaches (and never pays for) the model critic.
critic_never_called = model_critic(scripted(), rubric)   # empty script: would raise if invoked
assert len(combined(verify_sql, critic_never_called)(bad)) == 2
```

The ordering is deliberate: the tool verifier is deterministic, fast, and authoritative on correctness, so it goes first and short-circuits. The model critic is for what tools cannot measure, and it only runs on drafts that already work. This is also the cost-control answer: you pay for a model critique only when the draft has cleared the bar that does not cost anything.

## How LangGraph does it

Anthropic's "Building effective agents" calls this shape the **evaluator-optimizer** workflow: one call generates, another evaluates and gives feedback, loop. In LangGraph it is two nodes and a conditional edge, with the round counter in state so the bound is explicit.

```text
from scratch                          LangGraph
------------                          ---------
prompt list + best + trace            typed state (draft, issues, round, best)
generator.invoke(...)                 `generate` node
verifier(draft)                       `evaluate` node
if not issues / round < max           conditional edge after `evaluate`
for round_no in range(max_rounds)     the loop edge back to `generate`, bounded by state["round"]
```

```{code-cell} python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END


class ReflectState(TypedDict, total=False):
    task: str
    draft: str
    issues: list[str]
    round: int
    accepted: bool
    best: dict                                # fewest-issues draft so far
    history: Annotated[list, operator.add]    # append: every (draft, issues) pair, for the trace


def make_generate(model):
    def generate(state: ReflectState):
        prompt = [SystemMessage("Write one SQL query that answers the question. Return only SQL."),
                  HumanMessage(state["task"])]
        for past in state.get("history", []):    # the optimizer sees every prior attempt
            prompt += [AIMessage(past["draft"]), HumanMessage("Revise. Issues:\n- " + "\n- ".join(past["issues"]))]
        return {"draft": model.invoke(prompt).content.strip(), "round": state.get("round", 0) + 1}
    return generate


def evaluate(state: ReflectState):
    issues = verify_sql(state["draft"])
    best = state.get("best")
    if best is None or len(issues) < len(best["issues"]):
        best = {"draft": state["draft"], "issues": issues}
    return {"issues": issues, "accepted": not issues, "best": best,
            "history": [{"draft": state["draft"], "issues": issues}]}


def should_revise(state: ReflectState, max_rounds=3):
    if state["accepted"]:
        return END
    return "generate" if state["round"] < max_rounds else END


b = StateGraph(ReflectState)
b.add_node("generate", make_generate(scripted(
    "SELECT COUNT(*) FROM seats WHERE customer = 'Acme' AND active = 1",
    "SELECT COUNT(*) FROM seats WHERE customer = 'Acme' AND month = '2026-09' AND active = 1",
    "SELECT COUNT(DISTINCT user_id) FROM seats WHERE customer = 'Acme' AND month = '2026-09' AND active = 1",
)))
b.add_node("evaluate", evaluate)
b.add_edge(START, "generate")
b.add_edge("generate", "evaluate")
b.add_conditional_edges("evaluate", should_revise, ["generate", END])
graph = b.compile()

out = graph.invoke({"task": "How many Acme users had an active seat in September 2026?"})
print("rounds:", out["round"], "accepted:", out["accepted"])
print(out["best"]["draft"])
assert out["accepted"] and out["round"] == 3
assert [len(h["issues"]) for h in out["history"]] == [2, 1, 0]
```

```text
START ──► generate ──► evaluate ──(issues, round < 3)──► generate
                           │
                           ├──(no issues)──────────────► END  accepted
                           └──(issues, round == 3)─────► END  best-so-far
```

Underneath: `generate` is the generator call with the prompt rebuilt from `history`, which is the message list our loop appended to. `evaluate` is the verifier plus the best-so-far bookkeeping. The conditional edge is the two `if` statements. `history` uses an append reducer because we want every attempt kept; `draft` and `issues` use replace because there is one current draft. Put an `InMemorySaver` on this graph and every round is a checkpoint: a reflection loop that crashes on round 3 of a slow verifier resumes at round 3, and you can inspect the history of drafts for any thread after the fact, which is how you find out that your critic is oscillating.

### Where the verifier should live

In the graph above the verifier is a node. It could also be a **tool** the generating agent calls itself: give the agent `run_sql_check(query)` and let it decide to verify before answering. Both work; they differ in who controls the loop.

```text
verifier as node      your code guarantees the check runs; the model cannot skip it
verifier as tool      the model decides when to check; cheaper when it is confident, but skippable
```

For anything with consequences, the node wins. "The model chooses, your software executes" applies to verification too: if the check must happen, it is an edge in your graph, not a suggestion in your prompt. The current guidance from both Anthropic and OpenAI on agentic tasks is to define what "done" means and how the model should verify its work; the graph is where you make that definition enforceable.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model

real = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6"))
b = StateGraph(ReflectState)
b.add_node("generate", make_generate(real))
b.add_node("evaluate", evaluate)
b.add_edge(START, "generate")
b.add_edge("generate", "evaluate")
b.add_conditional_edges("evaluate", should_revise, ["generate", END])
live_graph = b.compile()
out = live_graph.invoke({"task": "How many Acme users had an active seat in September 2026? "
                                 "Table seats(customer, user_id, month 'YYYY-MM', active 0/1). Return only SQL."})
for h in out["history"]:
    print(len(h["issues"]), "issues:", h["draft"])
print("accepted:", out["accepted"])
```

## What goes wrong in production

1. **The loop never ends.** Symptom: one request runs 40 model calls and times out. Cause: no round bound, or a critic with a rubric the generator cannot satisfy (contradictory requirements). Fix: `max_rounds` in code, best-so-far return, and a check that consecutive rounds actually reduce the issue count; if round N has the same issues as round N-1, stop early, you are oscillating.

2. **Self-critique that rubber-stamps.** Symptom: every draft passes reflection on round 1 and the error rate is unchanged. Cause: the same model, same context, asked if it agrees with itself. Fix: separate the evaluator from the generator (fresh context, explicit rubric, ideally a different model), and put an executable verifier in front of any model critic wherever one can exist.

3. **Critiques the generator cannot act on.** Symptom: rounds go by with cosmetic changes. Cause: feedback like "improve accuracy." Fix: verifiers return one specific, located, actionable item per failure. Structural checks name the requirement and the fix. Model critics are given a rubric and told to cite the rubric item.

4. **Verifying the explanation instead of the artifact.** Symptom: the model's reasoning says "I filtered by month" and the query does not. Cause: the critic read the prose. Fix: verify the thing itself. Run the SQL. Run the tests. Diff the output. Chain-of-thought text is a signal about what the model attended to, not a record of what it did.

5. **Cost triples silently.** Symptom: the feature works and the bill is 3x. Cause: every request pays for generate plus critique plus regenerate, including the 80% that were fine the first time. Fix: cheap deterministic checks first and short-circuit; a model critic only on drafts that pass the tools; reflection only on the request classes where the eval shows it helps.

6. **The verifier is wrong.** Symptom: correct answers get rejected; the loop "fixes" them into wrong ones. Cause: a stale fixture, a brittle regex, an expected value computed by hand. Fix: the verifier is code, so it gets tests. Track the verifier's false-reject rate in evals. When the loop exhausts its budget, log the final issues so a human can see whether the critic or the generator was at fault.

7. **Revision without memory.** Symptom: round 3 resubmits round 1's draft. Cause: each revision prompt contained only the latest critique. Fix: the full history of drafts and issues stays in the prompt (or is summarized when long). That is what the `history` reducer is for.

## What changed by 2026

- **Evaluation and generation are separated by design.** Anthropic's harness-design guidance for long-running work names a Planner, a Generator, and an Evaluator as distinct roles, with gradable rubrics and "sprint contracts" that define done before the work starts, precisely because a generator grading itself is unreliable. Reflection in 2026 means an evaluator role, not a "please double-check" instruction.
- **Verifiers moved from prose to execution.** The default critic for code is the test suite; for SQL it is running the query against a fixture; for structured output it is the schema. Model critics remain for qualities tools cannot measure, and they run after the tools, with a rubric.
- **Reasoning models internalized part of the loop.** Models trained with reinforcement learning on verifiable outcomes exhibit self-checking behavior during generation, which reduces the value of an external "reflect on this" step for simple tasks. It does not remove the need for external verification, and the faithfulness research is the reason: what the model writes about its checking is not a reliable account of it. Verify the outcome.
- **Best-of-N with a verifier replaced self-consistency for many tasks.** Sampling several candidates and letting an executable verifier pick is a reflection loop with the rounds run in parallel (03. Parallelization). It is often cheaper in wall time and stronger than sequential self-revision.
- **Reflection became an eval question.** Whether to add a reflection step is now answered by running the eval suite with and without it and comparing quality against cost and latency. Teams remove reflection from request classes where it does not move the numbers; Anthropic's guidance to remove harness components as models improve applies directly.
- **Harnesses made "define done" operational.** OpenAI's reasoning guidance for agentic work says to define what counts as done and how the model should verify it; LangGraph's evaluator node and Deep Agents' subagent-based review are where that definition lives in code rather than in a prompt.

## Interview questions

**Q1. What is the reflection pattern and when does it help?**

Strong: Reflection is a generate, evaluate, revise loop: produce an artifact, have an evaluator check it against explicit criteria, feed the specific issues back, and regenerate, bounded by a round limit. It helps when the task has checkable criteria, when the first draft is often close but not right, and when the cost of a wrong output is higher than the cost of a second model call. It helps least when the criteria are vague or when the evaluator shares the generator's blind spots. I decide whether to keep it by running the eval suite with and without the loop and comparing quality against cost.

Weak: "Ask the model to review and improve its answer."

**Q2. Why is self-reflection weaker than external verification?**

Strong: Because the model that made the error made it for a reason it found convincing, and asking the same model in the same context whether it agrees with itself asks the same beliefs. It catches slips, not misconceptions. There is also a faithfulness problem: research on reasoning models shows their written reasoning often does not reflect what actually drove the answer, so "I checked and it is correct" is generated text, not evidence. An external verifier, a test suite, a database, a schema, a separate model with a rubric, checks the artifact itself. Anthropic's harness guidance separates generation from evaluation for exactly this reason.

Weak: "The model might miss things."

**Q3. How do you stop a reflection loop from running forever or costing too much?**

Strong: A hard round bound in code, not in the prompt. Best-so-far tracking so that budget exhaustion returns the draft with the fewest issues, flagged as not accepted, rather than the last draft. An oscillation check: if the issue set did not shrink between rounds, stop early. For cost, order the verifiers so cheap deterministic checks run first and short-circuit, and only drafts that pass them reach a model critic. And I only enable reflection for request classes where the eval shows it moves the numbers.

Weak: "Set a max iterations."

**Q4. How would you build the evaluator-optimizer pattern in LangGraph?**

Strong: Two nodes and a conditional edge. The generate node rebuilds its prompt from a history key in state that holds every previous draft and its issues, so the optimizer never resubmits a failed attempt. The evaluate node runs the verifier and updates issues, accepted, and best-so-far. The conditional edge returns END if accepted or the round count hit the bound, otherwise loops back to generate. History uses an append reducer; draft and issues use replace. With a checkpointer every round is persisted, so a crash resumes mid-loop and I can inspect the sequence of drafts to see whether the critic is oscillating.

Weak: "A loop between a writer node and a reviewer node."

**Q5. Should the verifier be a node in the graph or a tool the agent can call?**

Strong: If the check must happen, it is a node: my code guarantees the edge runs and the model cannot skip it. If the check is optional and the model is well calibrated about when it needs it, a tool is cheaper because it only runs when the model asks. For anything with consequences, refunds, schema changes, customer-facing claims, I make it a node. It is the same principle as tool execution: the model chooses, the software enforces. A prompt saying "always verify before answering" is a suggestion; an edge is a guarantee.

Weak: "Give the agent a verify tool."

**Q6. What makes a critique useful to the generator?**

Strong: It is specific, located, and actionable: it names the requirement that failed and what would satisfy it, one item per failure, and it lists every failure in the round, not just the first. "No date filter; the question asks about September; filter on month = '2026-09'" produces a fix. "Could be more accurate" produces a cosmetic rewrite and burns a round. For model critics I give a rubric and require each issue to cite a rubric item. And the full history of drafts and critiques stays in the prompt so the generator sees what was already tried.

Weak: "Tell the model what is wrong."

**Q7. Your reflection loop passes every draft on round one, but the error rate has not improved. What happened?**

Strong: The critic is rubber-stamping, almost certainly because it is the same model in the same context. I would first check what the critic actually receives and whether it can fail at all: give it known-bad drafts and measure its reject rate. Then I would replace or front it with an executable verifier wherever one can exist, and if I keep a model critic, run it in a fresh context with an explicit rubric and ideally a different model. I would also check that the pass criterion is not trivially satisfiable by the generator's default output. Finally I would compare eval numbers with the loop removed, because a loop that does not change outcomes is only a cost.

Weak: "Make the reviewer prompt stricter."

## Build it yourself

Extend the LangGraph evaluator-optimizer:

1. Add an oscillation check to `should_revise`: if the issue list in the latest `history` entry equals the previous one, go to `END` even if rounds remain, and record `stopped_reason` in state as `"oscillating"`, `"accepted"`, or `"budget"`.
2. Add a `verifier_cost` counter (an int with `operator.add`) that adds 1 for each `verify_sql` call and 10 for each model-critic call, and route to a `critic` node only when `verify_sql` returns no issues.
3. Add a best-of-N variant: a `sample` node that emits three `Send`s to `generate` with different seeds (script three different drafts), an `evaluate` per draft, and a `pick` node that selects the draft with the fewest issues.
4. Write a small test for the verifier itself: three known-good and three known-bad queries, asserting the false-reject and false-accept counts are zero.

Acceptance checks:

- A scripted generator that returns the same wrong query twice ends with `stopped_reason == "oscillating"` after exactly 2 rounds.
- The happy-path run ends with `verifier_cost == 13` (three tool checks, one model critique on the passing draft).
- The best-of-N variant returns the query with `COUNT(DISTINCT user_id)` and its `history` has exactly 3 entries.
- The verifier test passes, and deliberately breaking `EXPECTED` makes it fail on the good queries.

## Key takeaways

- Reflection is generate, evaluate, revise; its value is decided entirely by who the evaluator is.
- Self-reflection shares the generator's blind spots and its written self-checks are not evidence; verify the artifact, not the prose about it.
- The strongest critics execute: run the SQL, run the tests, check the schema. Model critics with rubrics come after, for what tools cannot measure.
- Critiques must be specific, located, actionable, and complete per round, or the loop burns iterations on cosmetic changes.
- Bound the loop in code: max rounds, best-so-far on exhaustion, and an oscillation check.
- In LangGraph the evaluator-optimizer is two nodes and a conditional edge, with history appended and the round count in state; a checkpointer makes every round inspectable and resumable.
- If a check must happen, it is a node, not a tool the model may call; and whether to keep the loop at all is an eval question.
