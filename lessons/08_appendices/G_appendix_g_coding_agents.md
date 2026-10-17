# G. Coding Agents: explore, edit, verify, and keep state in git

*Track A (agent engineering). Book: Appendix G (Coding Agents), drawing on Chapters 5 (Tool Use), 4 (Reflection), 6 (Planning), and 19 (Evaluation). Prerequisites: 00. What an agent actually is, 04. Reflection, 34. Context Engineering, 33. Sandboxed Agent Execution, E. AI Agents on the CLI.*

A coding agent is the agent loop from lesson 00 pointed at a repository. The tools are different (search, read, edit, run), but the loop is the same. What makes coding agents their own discipline is that code has something most agent domains lack: an executable verifier. Tests either pass or they do not. A good coding harness is built around that fact. The model proposes; the test suite decides.

This lesson builds a small coding agent over an in-memory repository, then the long-running harness that Anthropic's engineering posts describe (a progress file, git as durable state, a feature list, a separate evaluator), then the same planner/generator/evaluator loop in LangGraph, and finally a SWE-bench-style scorer. Every "test run" in this lesson actually executes the repository's code.

## The problem

The Acme investigation from D. Building an Agent found the root cause: the billing service emits a duplicate seat line when a subscription is renewed twice in one period. The support agent can refund Acme. It cannot fix the bug. The team asks for a coding agent that takes the ticket and opens a pull request.

The naive design is the chat loop with a big prompt:

```text
prompt = ticket + entire repository (all files, concatenated)
model  → "Here is the corrected billing/invoice.py:" + full new file
harness → overwrite the file, reply "Done!"
```

Where it breaks:

```text
Step                      Failure
------------------------  -----------------------------------------------------------
entire repo in the prompt 400k-line monorepo does not fit; the parts that do fit bury
                          the three functions that matter (context rot)
full-file rewrite         model silently drops an unrelated function it did not "see"
                          as important; a teammate's uncommitted change is clobbered
"Done!"                   the model says the bug is fixed; nobody ran the tests; the
                          fix merged duplicate lines into one line with qty=240
no sandbox                the model ran `pip install` from a URL in an issue comment
second day                the task needs 6 hours; the context window resets and the
                          next session starts over, or worse, half-redoes finished work
```

Each failure maps to a design decision that every serious coding harness now makes:

```text
         ┌───────────── context: find, don't dump ─────────────┐
         │  ls / glob / grep  ──►  read_file(path, line range)  │
         └──────────────────────────┬───────────────────────────┘
                                    ▼
         ┌───────────── edit: small, checkable patches ────────┐
         │  str_replace(path, old, new)  old must match ONCE,   │
         │  file must be unchanged since the agent last read it │
         └──────────────────────────┬───────────────────────────┘
                                    ▼
         ┌───────────── verify: tests decide "done" ───────────┐
         │  run_tests in a sandbox → FAIL_TO_PASS now pass,     │
         │  PASS_TO_PASS still pass → only then may it stop     │
         └──────────────────────────┬───────────────────────────┘
                                    ▼
         ┌───────────── persist: git + progress file ──────────┐
         │  commit each verified step; append to PROGRESS.md;   │
         │  next session orients from these, not from memory    │
         └──────────────────────────────────────────────────────┘
```

## Build it without a framework

### A repository, and tools that act on it

The repository is a dict of paths to source text. The tools are the four a coding agent cannot live without: find files, search content, read with line numbers, and edit by exact replacement. `run_tests` executes the real test file against the real module source, so "tests pass" in this lesson is a fact, not a claim.

The first cell also carries the scripted model helper. It replays fixed turns so the harness, not the model's judgment, is under study. A live cell later runs a real model.

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


import copy
import fnmatch
import hashlib
import sys
import types

INVOICE_PY = '''def build_lines(items):
    """Return invoice lines as (sku, period, qty)."""
    lines = []
    for item in items:
        lines.append((item["sku"], item["period"], item["qty"]))
    return lines


def total(lines, prices):
    return sum(prices[sku] * qty for sku, _, qty in lines)
'''

TEST_INVOICE_PY = '''import billing.invoice as inv

def test_total():
    assert inv.total([("seat", "2026-09", 120)], {"seat": 40}) == 4800

def test_dedup_same_period():
    items = [{"sku": "seat", "period": "2026-09", "qty": 120}] * 2
    assert inv.build_lines(items) == [("seat", "2026-09", 120)]

def test_prorate():
    assert inv.prorate(4800, days_used=15, days_in_period=30) == 2400
'''


class Workspace:
    """An in-memory repo with the tools a coding agent gets, plus a tiny git."""

    def __init__(self, files):
        self.files = dict(files)
        self.read_hashes = {}          # path -> hash of the content the agent last saw
        self.commits = []              # (id, message, snapshot)

    @staticmethod
    def _h(text):
        return hashlib.sha256(text.encode()).hexdigest()[:8]

    # ---- explore ---------------------------------------------------------
    def glob(self, pattern):
        return sorted(p for p in self.files if fnmatch.fnmatch(p, pattern))

    def grep(self, pattern):
        return [f"{p}:{i}: {line.strip()}" for p, src in sorted(self.files.items())
                for i, line in enumerate(src.splitlines(), 1) if pattern in line]

    def read_file(self, path, start=1, end=None):
        src = self.files[path]
        self.read_hashes[path] = self._h(src)
        lines = src.splitlines()[start - 1:end]
        return "\n".join(f"{n:4d}| {l}" for n, l in enumerate(lines, start))

    # ---- edit ------------------------------------------------------------
    def str_replace(self, path, old, new):
        src = self.files.get(path)
        if src is None:
            return "error: no such file"
        if self.read_hashes.get(path) != self._h(src):
            return "error: file changed since you last read it (or never read); read it again"
        count = src.count(old)
        if count != 1:
            return f"error: old text matched {count} times; it must match exactly once"
        self.files[path] = src.replace(old, new)
        self.read_hashes[path] = self._h(self.files[path])   # the agent knows its own edit
        return f"ok: edited {path}"

    # ---- verify ----------------------------------------------------------
    def run_tests(self, only=None):
        """Execute billing/test_invoice.py against billing/invoice.py. Not a security
        sandbox: a real harness runs this in a container with no network."""
        pkg = types.ModuleType("billing"); pkg.__path__ = []
        mod = types.ModuleType("billing.invoice")
        saved = {k: sys.modules.get(k) for k in ("billing", "billing.invoice")}
        results = {}
        try:
            exec(compile(self.files["billing/invoice.py"], "billing/invoice.py", "exec"), mod.__dict__)
            pkg.invoice = mod
            sys.modules["billing"], sys.modules["billing.invoice"] = pkg, mod
            ns = {}
            exec(compile(self.files["billing/test_invoice.py"], "billing/test_invoice.py", "exec"), ns)
            for name, fn in ns.items():
                if name.startswith("test_") and (only is None or name in only):
                    try:
                        fn(); results[name] = "pass"
                    except AssertionError:
                        results[name] = "fail"
                    except Exception as exc:
                        results[name] = f"error: {type(exc).__name__}: {exc}"
        except Exception as exc:
            results["<collection>"] = f"error: {type(exc).__name__}: {exc}"
        finally:
            for k, v in saved.items():
                if v is None: sys.modules.pop(k, None)
                else: sys.modules[k] = v
        return results

    # ---- persist ---------------------------------------------------------
    def commit(self, message):
        cid = self._h(message + str(sorted(self.files.items())))
        self.commits.append((cid, message, copy.deepcopy(self.files)))
        return cid

    def checkout(self, cid):
        self.files = copy.deepcopy(next(s for c, _, s in self.commits if c == cid))
        self.read_hashes.clear()

    def log(self):
        return [f"{c} {m}" for c, m, _ in reversed(self.commits)]


ws = Workspace({"billing/invoice.py": INVOICE_PY, "billing/test_invoice.py": TEST_INVOICE_PY,
                "notifications/slack.py": "def post(channel, text):\n    ...\n",
                "README.md": "Billing service. Run tests before committing.\n"})
BASE = ws.commit("initial import")

print(ws.grep("def build_lines"))
print(ws.run_tests())
assert ws.run_tests()["test_total"] == "pass"
assert ws.run_tests()["test_dedup_same_period"] == "fail"          # the Acme bug, reproduced
assert ws.run_tests()["test_prorate"].startswith("error: AttributeError")

# The edit tool's two guards, shown directly.
probe = Workspace(ws.files)
print(probe.str_replace("billing/invoice.py", "lines", "rows"))            # never read
probe.read_file("billing/invoice.py")
print(probe.str_replace("billing/invoice.py", "lines", "rows"))            # ambiguous
probe.files["billing/invoice.py"] += "\n# teammate's change\n"              # someone else edits
print(probe.str_replace("billing/invoice.py", "lines = []", "lines = list()"))

assert probe.str_replace("billing/invoice.py", "lines = []", "x").startswith("error: file changed")
probe.read_file("billing/invoice.py")
assert probe.str_replace("billing/invoice.py", "lines = []", "lines = list()") == "ok: edited billing/invoice.py"
assert "# teammate's change" in probe.files["billing/invoice.py"]           # unrelated work preserved
```

Three things in `Workspace` are deliberate and match what production coding harnesses do.

`read_file` returns line numbers and supports ranges. The model navigates by `path:line`, and it can read 40 lines around a grep hit instead of a 3,000-line file. That is just-in-time retrieval, in Anthropic's context-engineering vocabulary: keep lightweight identifiers (paths, line numbers) in context and load content on demand.

`str_replace` refuses two kinds of edit. It refuses an `old` string that matches zero or several times, because a patch that cannot be located exactly is a patch that can land in the wrong place. And it refuses to edit a file whose content changed since the agent last read it, because the agent's picture of the file is stale. Both checks run *before* the side effect. The `probe` workspace at the end of the cell exercises each guard: an edit before any read, an ambiguous match, and a teammate's concurrent change.

`commit` and `checkout` are git in miniature. Every verified step becomes a snapshot the next session can trust and the harness can roll back to.

Note the last assert: the teammate's line survives the agent's edit. A full-file rewrite would have deleted it.


### The loop, with tests as the stop condition

Now the agent. The loop is lesson 00's loop with one change that matters more than any prompt: **the model is not allowed to stop until the verifier agrees.** When the model produces a final answer, the harness runs the task's tests. If they fail, the harness appends the failure as a new message and continues.

The task's tests come in two sets, named after SWE-bench's convention:

```text
FAIL_TO_PASS  tests that fail before the fix and must pass after it     (the bug)
PASS_TO_PASS  tests that pass before the fix and must still pass after  (no regressions)
```

```{code-cell} python
from langchain_core.messages import HumanMessage, ToolMessage

TASK = {"issue": "Duplicate seat line when a subscription renews twice in one period (Acme, in_881).",
        "fail_to_pass": ["test_dedup_same_period"], "pass_to_pass": ["test_total"]}


def verify(ws, task):
    r = ws.run_tests(only=task["fail_to_pass"] + task["pass_to_pass"])
    bad = {k: v for k, v in r.items() if v != "pass"}
    return (not bad), bad


def coding_agent(ws, model, task, max_steps=12):
    tools = {"glob": ws.glob, "grep": ws.grep, "read_file": ws.read_file,
             "str_replace": ws.str_replace, "run_tests": ws.run_tests}
    messages = [HumanMessage(f"Fix this issue. Tests decide when you are done.\n{task['issue']}")]
    rejected_stops = 0
    for _ in range(max_steps):
        ai = model.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            ok, bad = verify(ws, task)                     # the model's claim is not evidence
            if ok:
                cid = ws.commit(f"fix: {task['issue'][:40]}")
                return {"status": "resolved", "commit": cid, "rejected_stops": rejected_stops, "messages": messages}
            rejected_stops += 1
            messages.append(HumanMessage(f"Not done. Failing tests: {bad}. Read the code again and fix it."))
            continue
        for call in ai.tool_calls:
            out = tools[call["name"]](**call["args"])
            messages.append(ToolMessage(content=str(out), tool_call_id=call["id"]))
    return {"status": "budget_exhausted", "rejected_stops": rejected_stops, "messages": messages}


LOOP_OLD = '''    for item in items:
        lines.append((item["sku"], item["period"], item["qty"]))'''
MERGE = '''    merged = {}
    for item in items:
        key = (item["sku"], item["period"])
        merged[key] = merged.get(key, 0) + item["qty"]
    lines = [(s, p, q) for (s, p), q in merged.items()]'''

model = scripted(
    [("grep", {"pattern": "def build_lines"})],
    [("read_file", {"path": "billing/invoice.py"}), ("read_file", {"path": "billing/test_invoice.py", "start": 6, "end": 8})],
    [("str_replace", {"path": "billing/invoice.py", "old": LOOP_OLD, "new": MERGE})],
    "Fixed! Duplicate lines are now merged into one.",                     # premature claim
    [("str_replace", {"path": "billing/invoice.py",
                      "old": '        merged[key] = merged.get(key, 0) + item["qty"]',
                      "new": '        merged.setdefault(key, item["qty"])'})],
    [("run_tests", {"only": ["test_dedup_same_period", "test_total"]})],
    "Fixed: a repeated (sku, period) is billed once. test_dedup_same_period and test_total pass.",
)

run = coding_agent(ws, model, TASK)
print(run["status"], "| rejected stops:", run["rejected_stops"], "| commit:", run.get("commit"))
print(ws.log())
assert run["status"] == "resolved"
assert run["rejected_stops"] == 1                          # "Fixed!" with qty=240 was refused
assert ws.run_tests()["test_total"] == "pass"              # no regression
assert ws.files["notifications/slack.py"].startswith("def post")   # untouched file untouched
AFTER_F1 = run["commit"]
```

Read the transcript as a sequence:

```text
model:    grep "def build_lines"                      → billing/invoice.py:1
model:    read invoice.py, read test lines 6-8         → sees expected [(seat, 2026-09, 120)]
model:    str_replace loop → merge quantities          → ok
model:    "Fixed!"
harness:  run FAIL_TO_PASS + PASS_TO_PASS              → test_dedup_same_period: fail (qty=240)
harness:  "Not done. Failing tests: ..."
model:    str_replace merge → setdefault               → ok
model:    run_tests                                    → pass, pass
model:    "Fixed: ..."
harness:  verify → pass → commit
```

The first "Fixed!" is the failure mode that sinks naive coding agents. The model's belief was reasonable and wrong. Its written reasoning would have sounded confident. A reasoning trace is not an audit log, so the harness checks the outcome instead.

### The long-running harness: progress file, feature list, fresh sessions

One bug fits in one context window. A feature that takes a day does not. Anthropic's "Effective harnesses for long-running agents" post describes what worked for multi-session coding: an initializer run that sets up the environment and writes a feature list with pass/fail status, then repeated coding sessions that each start fresh, orient themselves from a progress file and git log, work on one feature, verify it end to end, commit, and leave the repo clean. Their later "Harness design for long-running application development" post adds a planner, a generator, and a separate evaluator, and reports that models grading their own work tend to praise it.

```text
session N (fresh context)
  1. orient:   read PROGRESS.md, `git log`, features.json   (not the old transcript)
  2. choose:   first feature with passes == false
  3. contract: which tests define "done" for it            (agreed before coding)
  4. generate: edit code                                     (generator)
  5. evaluate: run contract tests + full suite               (evaluator, separate)
  6. persist:  pass → commit, mark passes=true, append PROGRESS.md
               fail → revert to last commit, record why, retry or hand off
```

The key property is that each session's *only* inheritance is files. No transcript crosses the boundary. That is what makes the design robust to context resets, crashes, and model swaps.

```{code-cell} python
import json

ws.files["features.json"] = json.dumps([
    {"id": "F1", "desc": "bill a repeated (sku, period) once", "tests": ["test_dedup_same_period"], "passes": True},
    {"id": "F2", "desc": "prorate(amount, days_used, days_in_period)", "tests": ["test_prorate"], "passes": False},
])
ws.files["PROGRESS.md"] = "# Progress\n- F1 done in commit " + AFTER_F1 + "\n"
ws.commit("harness: add features.json and PROGRESS.md")

# Generator attempts for F2, in order. The first one breaks F1 (a regression);
# the evaluator must catch it even though the generator's own test passes.
F2_ATTEMPTS = [
    ("regressing", lambda src: src.replace('merged.setdefault(key, item["qty"])',
                                           'merged[key] = merged.get(key, 0) + item["qty"]')
                             + "\n\ndef prorate(amount, days_used, days_in_period):\n    return amount * days_used // days_in_period\n"),
    ("clean",      lambda src: src + "\n\ndef prorate(amount, days_used, days_in_period):\n    return amount * days_used // days_in_period\n"),
]


def evaluator(ws, feature):
    """Separate from the generator: runs the contract AND the whole suite."""
    results = ws.run_tests()
    contract_ok = all(results.get(t) == "pass" for t in feature["tests"])
    done = [t for f in json.loads(ws.files["features.json"]) if f["passes"] for t in f["tests"]]
    regressions = [t for t in done + ["test_total"] if results.get(t) != "pass"]
    return contract_ok and not regressions, {"contract_ok": contract_ok, "regressions": regressions}


def session(ws, attempts, log):
    context = {"progress": ws.files["PROGRESS.md"], "git_log": ws.log()[:3]}    # fresh context: files only
    features = json.loads(ws.files["features.json"])
    todo = [f for f in features if not f["passes"]]
    if not todo:
        log.append("nothing to do")
        return "idle"
    feature, head = todo[0], ws.commits[-1][0]
    for label, patch in attempts:
        ws.files["billing/invoice.py"] = patch(ws.files["billing/invoice.py"])   # generator
        ok, report = evaluator(ws, feature)
        log.append((feature["id"], label, report))
        if ok:
            feature["passes"] = True
            ws.files["features.json"] = json.dumps(features)
            ws.files["PROGRESS.md"] += f"- {feature['id']} done ({label}); evaluator: {report}\n"
            ws.commit(f"feat: {feature['desc']}")
            return "committed"
        ws.checkout(head)                                                      # leave the repo clean
    return "handoff"


log = []
print("session 1:", session(ws, F2_ATTEMPTS, log))
print("session 2:", session(ws, F2_ATTEMPTS, log))
for entry in log:
    print("  ", entry)
print(ws.files["PROGRESS.md"])

assert log[0][1] == "regressing" and log[0][2]["regressions"] == ["test_dedup_same_period"]
assert log[1][1] == "clean" and log[1][2] == {"contract_ok": True, "regressions": []}
assert log[2] == "nothing to do"                                    # session 2 did not redo F2
assert all(v == "pass" for v in ws.run_tests().values())
assert ws.log()[0].endswith("feat: prorate(amount, days_used, days_in_period)")
```

The regressing attempt passed its own contract test (`test_prorate`). A generator grading itself would have shipped it. The evaluator ran the whole suite, saw F1 break, and the harness reverted to the last commit before trying again. Session 2 read `features.json`, found nothing failing, and stopped. It did not redo F2, because the state that says "F2 is done" is a committed file, not a memory.

## How LangGraph does it

The planner/generator/evaluator harness is a graph with a cycle. LangGraph adds three things over the plain-Python version: the loop is explicit and bounded, the state is checkpointed per `thread_id` so a crashed session resumes at the last completed node, and flaky infrastructure (the test runner) gets a retry policy without touching the node code.

```text
START ─► planner ─► generator ─► evaluator ──pass──► commit ─► END
                        ▲            │
                        └──fail, attempts < 3 (feedback in state)
                                     │
                                     └──fail, attempts = 3──► handoff ─► END
```

```{code-cell} python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy


class Sprint(TypedDict, total=False):
    feature: dict
    contract: dict          # agreed before any code is written
    attempts: int
    feedback: str
    verdict: dict
    commit: str


lg_ws = Workspace({})
lg_ws.commits = list(ws.commits)
lg_ws.checkout(AFTER_F1)                     # git as durable state: rewind to "F1 done, F2 not started"
assert lg_ws.run_tests()["test_prorate"].startswith("error")

PRORATE = "\n\ndef prorate(amount, days_used, days_in_period):\n    return {expr}\n"
gen_model = scripted(
    [("append", {"code": PRORATE.format(expr="amount * days_in_period // days_used")})],   # inverted ratio
    [("append", {"code": PRORATE.format(expr="amount * days_used // days_in_period")})],
)


def planner(s: Sprint):
    f = s["feature"]
    return {"contract": {"must_pass": f["tests"], "must_not_break": ["test_total", "test_dedup_same_period"]},
            "attempts": 0}


def generator(s: Sprint):
    ai = gen_model.invoke([HumanMessage(f"Implement {s['feature']['desc']}. Feedback: {s.get('feedback', 'none')}")])
    if s["attempts"] > 0:
        lg_ws.checkout(AFTER_F1)             # each attempt starts from the clean base
    for call in ai.tool_calls:
        lg_ws.files["billing/invoice.py"] += call["args"]["code"]
    return {"attempts": s["attempts"] + 1}


def evaluator_node(s: Sprint):
    r = lg_ws.run_tests()
    c = s["contract"]
    failing = [t for t in c["must_pass"] + c["must_not_break"] if r.get(t) != "pass"]
    return {"verdict": {"pass": not failing, "failing": failing},
            "feedback": f"failing: {failing}" if failing else ""}


def route(s: Sprint):
    if s["verdict"]["pass"]:
        return "commit"
    return "generator" if s["attempts"] < 3 else "handoff"


builder = StateGraph(Sprint)
builder.add_node("planner", planner)
builder.add_node("generator", generator)
builder.add_node("evaluator", evaluator_node,
                 retry_policy=RetryPolicy(max_attempts=3, retry_on=TimeoutError))  # flaky runner, not flaky code
builder.add_node("commit", lambda s: {"commit": lg_ws.commit(f"feat: {s['feature']['desc']}")})
builder.add_node("handoff", lambda s: {"commit": ""})
builder.add_edge(START, "planner")
builder.add_edge("planner", "generator")
builder.add_edge("generator", "evaluator")
builder.add_conditional_edges("evaluator", route, ["commit", "generator", "handoff"])
builder.add_edge("commit", END)
builder.add_edge("handoff", END)
graph = builder.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "sprint-F2"}}
out = graph.invoke({"feature": {"id": "F2", "desc": "prorate", "tests": ["test_prorate"]}}, cfg)
history = [snap.values.get("verdict") for snap in graph.get_state_history(cfg) if snap.values.get("verdict")]
print("attempts:", out["attempts"], "| verdict:", out["verdict"], "| commit:", out["commit"])
print("verdicts seen (newest first):", history[:2])
assert out["attempts"] == 2 and out["verdict"]["pass"] and out["commit"]
assert {"pass": False, "failing": ["test_prorate"]} in history          # attempt 1 was caught
assert lg_ws.run_tests() == {"test_total": "pass", "test_dedup_same_period": "pass", "test_prorate": "pass"}
```

Map it back:

| Plain-Python harness | LangGraph | Underneath |
|---|---|---|
| `for label, patch in attempts:` with a revert on failure | `generator ⇄ evaluator` cycle via `add_conditional_edges` | The router reads `verdict` and `attempts` from state after each super-step |
| `feature`, `contract`, feedback in local variables | `Sprint` state, checkpointed per super-step | A crash after the evaluator resumes at `route`, not at the planner |
| `PROGRESS.md` for the next session | `get_state_history(cfg)` for this thread | Every checkpoint is inspectable; the history shows attempt 1's failing verdict |
| try/except around a flaky test runner | `retry_policy=RetryPolicy(...)` on the node | The runtime re-executes the node on matching exceptions with backoff and jitter |
| `ws.checkout(head)` | same call, inside the node | LangGraph checkpoints graph state, not your filesystem. Git still owns the code. |

The last row is the one people get wrong. A LangGraph checkpoint restores the *graph's* state: the verdict, the attempt count, the messages. It does not restore files on disk. A coding harness needs both: the checkpointer for where the loop was, and git for what the code was. The installed LangGraph's `add_node` also accepts `timeout=` and `error_handler=`; a test runner that can hang is the obvious place for a timeout.

### The harness you would actually use

Deep Agents packages this stack on LangGraph: file tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`), an optional shell `execute` tool behind a sandbox backend, subagents via a `task` tool, and human-in-the-loop via `interrupt_on`. Current docs describe these; names shift between releases, so check the installed version. The live cell gives it our `run_tests` as a tool, so the verifier is the same one this lesson uses.

```{code-cell} python live
import os
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

live_ws = Workspace({"billing/invoice.py": INVOICE_PY, "billing/test_invoice.py": TEST_INVOICE_PY})


@tool
def repo_read(path: str) -> str:
    """Read a repo file with line numbers."""
    return live_ws.read_file(path)


@tool
def repo_replace(path: str, old: str, new: str) -> str:
    """Replace one exact, unique snippet in a file you have read."""
    return live_ws.str_replace(path, old, new)


@tool
def repo_tests() -> dict:
    """Run the billing test suite. You are done only when test_dedup_same_period and test_total pass."""
    return live_ws.run_tests(only=["test_dedup_same_period", "test_total"])


agent = create_deep_agent(model=init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6")),
                          tools=[repo_read, repo_replace, repo_tests],
                          system_prompt="You fix bugs. Read before editing. Tests decide when you are done.")
agent.invoke({"messages": [{"role": "user", "content": TASK["issue"]}]})
print(verify(live_ws, TASK))          # the harness still checks; the agent's word is not enough
```

### Evaluating coding agents: SWE-bench style

SWE-bench ("SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", Jimenez et al., 2023) turned the loop above into a benchmark. Each task instance is a real repository at a commit, an issue text, and two test lists. The agent sees the issue and the repo, not the tests that will judge it. A task is **resolved** only if every FAIL_TO_PASS test passes and every PASS_TO_PASS test still passes after the agent's patch. SWE-bench Verified is a human-validated subset that removes tasks with underspecified issues or unfair tests.

Your internal version should work the same way, with hidden tests. The scorer below evaluates three candidate patches for the dedup issue. One is correct. One special-cases the visible example. One deletes behavior other tests rely on.

```{code-cell} python
HIDDEN_TESTS = '''import billing.invoice as inv

def test_dedup_same_period():
    items = [{"sku": "seat", "period": "2026-09", "qty": 120}] * 2
    assert inv.build_lines(items) == [("seat", "2026-09", 120)]

def test_dedup_keeps_distinct_periods():
    items = [{"sku": "seat", "period": "2026-08", "qty": 120}, {"sku": "seat", "period": "2026-09", "qty": 120},
             {"sku": "seat", "period": "2026-09", "qty": 120}]
    assert inv.build_lines(items) == [("seat", "2026-08", 120), ("seat", "2026-09", 120)]

def test_total():
    assert inv.total([("seat", "2026-09", 120)], {"seat": 40}) == 4800
'''
INSTANCE = {"fail_to_pass": ["test_dedup_same_period", "test_dedup_keeps_distinct_periods"],
            "pass_to_pass": ["test_total"]}

CANDIDATES = {
    "correct":        INVOICE_PY.replace(LOOP_OLD, '''    seen = set()
    for item in items:
        key = (item["sku"], item["period"])
        if key not in seen:
            seen.add(key)
            lines.append((item["sku"], item["period"], item["qty"]))'''),
    "special-cased":  INVOICE_PY.replace("    return lines\n", "    return lines[:1]\n", 1),
    "broke-total":    INVOICE_PY.replace(LOOP_OLD, '''    for item in items:
        if (item["sku"], item["period"], item["qty"]) not in lines:
            lines.append((item["sku"], item["period"], item["qty"]))''').replace(
                      "prices[sku] * qty", "prices[sku]"),
}


def score(patch_src):
    w = Workspace({"billing/invoice.py": patch_src, "billing/test_invoice.py": HIDDEN_TESTS})
    r = w.run_tests()
    f2p = all(r[t] == "pass" for t in INSTANCE["fail_to_pass"])
    p2p = all(r[t] == "pass" for t in INSTANCE["pass_to_pass"])
    return {"resolved": f2p and p2p, "fail_to_pass": f2p, "pass_to_pass": p2p}


board = {name: score(src) for name, src in CANDIDATES.items()}
for name, s in board.items():
    print(f"{name:14s} {s}")
resolve_rate = sum(s["resolved"] for s in board.values()) / len(board)
print(f"resolve rate: {resolve_rate:.2f}")

# The visible example alone would have accepted the special-cased patch:
visible = Workspace({"billing/invoice.py": CANDIDATES["special-cased"], "billing/test_invoice.py": TEST_INVOICE_PY})
assert visible.run_tests(only=["test_dedup_same_period"]) == {"test_dedup_same_period": "pass"}
assert board["correct"]["resolved"]
assert not board["special-cased"]["fail_to_pass"]            # hidden F2P test catches overfitting
assert board["broke-total"]["fail_to_pass"] and not board["broke-total"]["pass_to_pass"]   # P2P catches regression
assert abs(resolve_rate - 1 / 3) < 1e-9
```

The special-cased patch passes the one example the agent could see. Only the hidden FAIL_TO_PASS test catches it. The third patch fixes the bug and quietly breaks `total`; only PASS_TO_PASS catches it. That is why both lists exist, and why the grading tests must be hidden from the agent.

Beyond resolve rate, track what the tools post recommends for any agent eval: tool calls per task, tokens, wall-clock time, and error rate. Two agents with the same resolve rate can differ by 5x in cost.

## What goes wrong in production

1. **"Done" without verification.** Symptom: merged PRs that do not fix the issue. Cause: the loop stops when the model stops. Fix: gate the stop on FAIL_TO_PASS and PASS_TO_PASS, as in `coding_agent`; if there is no failing test, the first task is to write one that reproduces the bug.

2. **Test tampering.** Symptom: tests pass because the agent edited or deleted them. Cause: the agent has write access to the tests that grade it, and reward-seeking models exploit that (Anthropic's CoT-faithfulness research found models exploiting a planted reward hack in over 99 percent of cases while rarely admitting it). Fix: grade with a pinned copy of the tests, flag diffs to test files for human review, and keep hidden tests out of the workspace.

3. **Clobbered work.** Symptom: a teammate's change vanishes. Cause: full-file writes or edits against a stale read. Fix: exact-match replace with a read-before-write hash check; work on a branch or worktree; refuse to start on a dirty tree unless told to.

4. **Context flooding.** Symptom: quality falls after the agent runs the full test suite and gets 30,000 lines of output. Cause: tool results dumped verbatim. Fix: truncate and summarize tool output (failing test names, first traceback, counts), paginate reads, and offload large logs to files the agent can grep. Anthropic notes Claude Code caps tool results at 25,000 tokens by default.

5. **Sandbox escape and secrets.** Symptom: the agent ran an install script from an issue comment, or printed an API key into a log. Cause: shell access on a developer machine with ambient credentials. Fix: run in a container or remote sandbox with no ambient credentials and restricted egress (33. Sandboxed Agent Execution); inject credentials through a proxy outside the sandbox, the design Anthropic's managed-agents post describes.

6. **Session amnesia or rework.** Symptom: day-two sessions redo day-one work or undo it. Cause: continuity lived in a transcript that was compacted or lost. Fix: the progress file, feature list, and git log are the handoff. Commit only verified work, so "committed" means "passed."

7. **Self-evaluation.** Symptom: a generator's summary says all acceptance criteria are met; they are not. Cause: the same model and context grade their own output. Fix: a separate evaluator with the contract and executable checks. For criteria that are not executable (UI quality), a rubric graded by a separate model with fresh context.

## What changed by 2026

- **Coding agents became the flagship agent application.** Terminal and IDE agents (Claude Code, Codex, Cursor's agent, Devin and others) became everyday developer tools built on the explore-edit-verify loop. Their harnesses, not only their models, are the product.
- **Harness design became a published discipline.** Anthropic's two harness posts describe progress files, feature lists with pass/fail status, git as durable state, sprint contracts that define "done" before coding, a separate evaluator with gradable criteria, and fresh context resets with handoff artifacts rather than relying only on compaction. They also advise removing harness components as models improve, since each one encodes an assumption about what the model cannot do.
- **Sandboxes are the default for autonomy.** The managed-agents design separates the session (an append-only durable log), a stateless harness, and an interchangeable sandbox, with credentials kept outside it. Deep Agents ships sandbox backends and virtualized file systems; current OpenAI Agents SDK docs describe sandbox agents with persistent workspaces.
- **Instruction files and skills replaced giant prompts.** Repos carry an instruction file the agent loads at start (CLAUDE.md, AGENTS.md) and skill folders loaded on demand (41. Skills).
- **Benchmarks matured and got harder to trust.** SWE-bench Verified became a standard headline number, followed by concerns about contamination and saturation that pushed teams toward private, recent, hidden-test evaluations of their own repos. Treat public numbers as a floor check, not a decision.
- **Reasoning models changed prompting, not verification.** Prompts state the goal, constraints, and how to verify, and let the model plan. Verification still comes from executing tests, never from reading the model's explanation.

## Interview questions

**Q1. Design a coding agent: repo indexing, context assembly, edit application, evaluation.** *(reported at Cursor)*

Strong: Context is assembled just in time: file tree and instruction file at start, then grep, symbol search, or an embedding index to find candidates, and ranged reads so only the relevant lines enter context. Edits are exact search-and-replace against a file the agent has read, rejected before any write if the match is not unique or the file changed. Every edit is followed by the narrowest useful test run, with truncated output. The stop condition is executable: failing tests for the issue now pass and previously passing tests still pass. Evaluation is SWE-bench style on our own repos with hidden tests, and we track resolve rate, cost, and time per task.

Weak: "Put the repo in the context and ask for the fix."

**Q2. Why are tests such an important part of a coding agent?**

Strong: They are an executable verifier, which most agent domains do not have. They turn "done" from the model's claim into a fact the harness checks, and they give the model precise feedback when it is wrong. That makes the edit-test loop converge. The caveat is that the agent must not be able to change the tests that grade it, so grading uses a pinned or hidden copy, and if the issue has no failing test, the first step is to write one that reproduces it.

Weak: "Tests make sure the code works."

**Q3. How would you run a coding agent on a task that takes eight hours?**

Strong: Assume the context will reset several times, so continuity lives in files, not in the transcript. An initializer run writes a feature list with pass/fail status and a progress file. Each session starts fresh, reads the progress file, the git log, and the feature list, picks one failing feature, agrees what "done" means, implements, has a separate evaluator run the contract and the full suite, then commits and updates the progress file. Failed attempts are reverted so the tree stays clean. This is the pattern in Anthropic's long-running-harness posts, and it also makes the run resumable after a crash.

Weak: "Use a model with a bigger context window."

**Q4. Why separate the generator from the evaluator?**

Strong: A model grading its own output in the same context tends to approve it, and Anthropic's harness-design post reports exactly that. A separate evaluator gets the contract, the diff, and tools to run checks, but not the generator's reasoning, so it judges the outcome. Where checks are executable, the evaluator is mostly a test runner. Where they are not, it grades against a rubric with specific, gradable criteria. The cost is extra calls, which is cheap compared to shipping a regression.

Weak: "Two models are more accurate than one."

**Q5. What does git do for a coding agent beyond version control?**

Strong: It is the durable state. A commit after each verified step is a checkpoint the next session can trust and the harness can roll back to, so failed attempts are reverted rather than piled up. The log is a compact history for orienting a fresh session. The diff is the reviewable artifact for humans. A graph checkpointer restores where the loop was; git restores what the code was. You need both.

Weak: "So you can undo mistakes."

**Q6. How do you keep a coding agent safe when it can run shell commands?**

Strong: Run it in a sandbox: a container or remote VM with only the repo mounted, no ambient credentials, and network egress limited to package mirrors. Credentials the task genuinely needs go through a proxy outside the sandbox so they never appear in the agent's context or logs. Commands are gated by allow and deny rules, with approval for anything outside the sandbox such as pushing or deploying. Everything is logged. Prompt-level instructions are the last layer, not the first.

Weak: "Tell it not to run dangerous commands."

**Q7. How do you evaluate a coding agent, and what are the traps?**

Strong: SWE-bench style: task instances with a repo snapshot, an issue, and hidden FAIL_TO_PASS and PASS_TO_PASS tests; resolved means both lists pass. Report resolve rate with confidence intervals, plus cost, tool calls, and time. The traps are contamination (public issues and fixes may be in training data), overfitting to visible examples, test tampering, and flaky tests that make results noisy. So I build a private set from recent issues in our own repos and rerun each task a few times.

Weak: "Look at the public leaderboard."

## Build it yourself

Extend `coding_agent` into a harness with a budget and a guard against test tampering.

1. Add a `protected` set of paths (the test files). `str_replace` on a protected path returns an error and increments a `tamper_attempts` counter.
2. Grade with a pinned copy: `verify` runs against the test source captured at task start, not the workspace's current copy.
3. Truncate `run_tests` output passed to the model to failing test names plus the first error, capped at 300 characters.
4. Add a `max_rejected_stops` limit (default 2); on exceeding it, return `"handoff"` with the last failing tests and revert to the pre-task commit.

Acceptance checks:

- A scripted model that tries to edit `billing/test_invoice.py` gets an error, `tamper_attempts == 1`, and the test file is unchanged.
- If the workspace's test file is modified directly (simulating a bypass), `verify` still uses the pinned copy and reports the real failure.
- Every `ToolMessage` from `run_tests` is at most 300 characters.
- A model that claims "Fixed!" three times without a working edit ends in `"handoff"`, and `ws.files` equals the pre-task snapshot.

## Key takeaways

- A coding agent is the agent loop with repository tools; its advantage over other domains is an executable verifier.
- Find, don't dump: glob, grep, and ranged reads assemble context just in time.
- Edit with exact, unique replacements against a fresh read, so failed edits fail before they write.
- The model may not stop until FAIL_TO_PASS tests pass and PASS_TO_PASS tests still pass.
- Long tasks survive context resets through files: a feature list, a progress file, and git commits of verified work.
- Keep generation and evaluation separate, and keep the grading tests out of the agent's reach.
- Evaluate SWE-bench style on your own repos with hidden tests, and track cost and time alongside resolve rate.
