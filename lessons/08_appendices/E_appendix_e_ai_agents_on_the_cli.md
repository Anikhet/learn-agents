# E. AI Agents on the CLI: streams, exit codes, permissions, and headless runs

*Track A (agent engineering) with a Track D (infra) flavor. Book: Appendix E (AI Agents on the CLI). Prerequisites: 00. What an agent actually is, 13. Human in the Loop, 12. Exception Handling and Recovery, D. Building an Agent.*

A command-line agent is an agent loop with a Unix contract wrapped around it. The loop is the same one you built in lesson 00. The contract is old: arguments in, stdin in, stdout for results, stderr for humans, an exit code for the caller. Most CLI agents that fail in automation do not fail in the loop. They fail the contract: they print chatty progress into the stream a script is parsing, they block on a permission prompt no one can answer, or they exit 0 after doing nothing.

This lesson builds a CLI for the running enterprise support/ops agent, first in plain Python, then on a LangGraph graph whose checkpointer turns "needs approval" from a hang into a resumable exit. Claude Code is the reference design throughout, because it is the CLI agent most engineers have used and its headless mode shows the whole contract.

## The problem

The ops team likes the support agent. Now they want to run it from places without a chat window:

```text
cron (02:00)      supportctl investigate --account acme --output-format json | jq .answer
CI pipeline       supportctl triage < ticket.txt
on-call laptop    supportctl investigate --account acme          (a human is watching)
another agent     subprocess.run(["supportctl", ...]) and parse the result
```

The first version is the chat loop with `print` and `input` sprinkled in:

```python
print("Looking up Acme in the CRM...")          # progress
print(answer)                                     # result
ok = input("Refund $4,800 to Acme? [y/N] ")       # permission
sys.exit(0)
```

Every line breaks one caller:

```text
Caller          What breaks                               Why
--------------  ----------------------------------------  --------------------------------
cron | jq       jq: parse error at line 1                 progress text is on stdout
CI              job hangs until the 6-hour timeout        input() waits on a closed stdin
another agent   treats a failed run as success            exit code is always 0
on-call human   cannot tell a slow lookup from a hang     no progress until the end
everyone        cannot resume after approving later       state died with the process
```

The fix is not smarter prompting. It is separating the channels, as every well-behaved Unix tool does:

```text
                   ┌──────────────────────────────┐
 argv, flags ─────►│                              │────► stdout: the RESULT (text, JSON,
 stdin (task) ────►│        agent CLI             │      or one JSON event per line)
 env (keys, cfg) ─►│  loop + permission policy    │────► stderr: progress, logs, warnings
                   │  + budget + checkpointer     │────► exit code: 0 ok, non-zero = why not
 TTY? ────────────►│                              │────► state on disk: resumable thread
                   └──────────────────────────────┘
```

Six design decisions follow from that picture, and they are the checklist for this lesson:

1. **Inputs.** Task from an argument or from stdin (`-`), config from flags and environment, never from an interactive question in headless mode.
2. **Output modes.** Human text by default; `json` for a single final object; `stream-json` for newline-delimited events a parent process can consume incrementally.
3. **Progress.** On stderr, or as typed events in `stream-json`. Never mixed into a parseable stdout.
4. **Exit codes.** Distinct codes for success, agent failure, usage error, approval required, budget exhausted, retryable failure, and interruption.
5. **Permissions.** An explicit policy per tool: allow, deny, or ask. "Ask" must have a defined meaning when nobody can answer.
6. **Non-interactive mode.** Detected (no TTY) or declared (`--non-interactive`), and it changes behavior: no prompts, fail closed, persist state, print how to resume.

## Build it without a framework

Start with the model and tools. Every offline cell uses a scripted stand-in for the model: it replays fixed turns so the CLI contract, not the model's judgment, is under study. A live cell at the end runs the same CLI against a real model.

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


from langchain_core.tools import tool

REFUNDS_ISSUED = []   # the side-effect ledger; tests assert on it


@tool
def crm_lookup(account: str) -> str:
    """Look up an account in the CRM: plan, owner, open tickets."""
    return f"{account}: plan=Enterprise seats=120 owner=j.lee open_tickets=2"


@tool
def stripe_get_invoice(account: str) -> str:
    """Fetch the latest Stripe invoice for an account (read-only)."""
    return f"{account}: invoice in_881 amount=$9,600 lines=[seats x120, seats x120 DUPLICATE]"


@tool
def stripe_refund(invoice_id: str, amount_usd: int) -> str:
    """Refund part of a Stripe invoice. Moves money; requires approval."""
    REFUNDS_ISSUED.append((invoice_id, amount_usd))
    return f"refunded ${amount_usd} on {invoice_id}"


TOOLS = {t.name: t for t in [crm_lookup, stripe_get_invoice, stripe_refund]}


def acme_script():
    """The run the ops team keeps asking for: investigate, then propose a refund."""
    return scripted(
        [("crm_lookup", {"account": "acme"}), ("stripe_get_invoice", {"account": "acme"})],
        [("stripe_refund", {"invoice_id": "in_881", "amount_usd": 4800})],
        "Acme was billed twice for 120 seats on in_881. Refund of $4,800 handled per policy.",
    )


print(sorted(TOOLS))
assert "stripe_refund" in TOOLS
```

### A permission policy, and what "ask" means with nobody there

Claude Code's permission model is the useful mental template: rules that allow, deny, or ask per tool, plus a mode that changes the default. Our version is a small function. The important branch is the last one: in non-interactive mode, "ask" cannot block, so it resolves to a *pending* decision that the CLI reports through its exit code.

```{code-cell} python
import fnmatch
import json
from dataclasses import dataclass, field


@dataclass
class PermissionPolicy:
    allow: list = field(default_factory=lambda: ["crm_*", "stripe_get_*"])   # read-only
    deny: list = field(default_factory=lambda: ["*_delete*"])
    ask: list = field(default_factory=lambda: ["stripe_refund"])             # money moves

    def decide(self, tool_name: str, interactive: bool, approver=None) -> str:
        """Return 'allow', 'deny', or 'pending'. Deny rules win over everything."""
        if any(fnmatch.fnmatch(tool_name, p) for p in self.deny):
            return "deny"
        if any(fnmatch.fnmatch(tool_name, p) for p in self.allow):
            return "allow"
        if any(fnmatch.fnmatch(tool_name, p) for p in self.ask):
            if interactive and approver is not None:
                return "allow" if approver(tool_name) else "deny"
            return "pending"        # headless: never block, never silently allow
        return "deny"               # unknown tools fail closed


class Events:
    """One sink for everything the CLI says. Mode decides which channel gets what."""

    def __init__(self, mode, stdout, stderr):
        self.mode, self.stdout, self.stderr = mode, stdout, stderr

    def emit(self, type_, **data):
        if self.mode == "stream-json":
            self.stdout.write(json.dumps({"type": type_, **data}) + "\n")   # machine channel
        elif type_ != "result":
            self.stderr.write(f"[{type_}] {data}\n")                        # human progress


policy = PermissionPolicy()
print({t: policy.decide(t, interactive=False) for t in ["crm_lookup", "stripe_refund", "sql_delete_rows", "web_fetch"]})
assert policy.decide("stripe_refund", interactive=False) == "pending"
assert policy.decide("stripe_refund", interactive=True, approver=lambda _: True) == "allow"
assert policy.decide("sql_delete_rows", interactive=True, approver=lambda _: True) == "deny"
assert policy.decide("web_fetch", interactive=False) == "deny"
```

Three choices in that code are the ones interviewers push on. Deny beats allow, so a broad allow rule cannot unlock a dangerous tool. Unknown tools fail closed, so a new tool added by an MCP server does not become callable by accident. And "pending" is a distinct outcome, not a disguised "deny": the caller needs to know the run stopped because it needs a human, not because it failed.

### The CLI itself: channels, modes, exit codes

The `main` function takes `argv`, `stdin`, `stdout`, `stderr`, and a TTY flag as parameters instead of reading globals. That is the standard trick for testing CLIs, and it is also what makes the channel separation visible.

```text
Exit code  Meaning                        Caller's reaction
---------  -----------------------------  -----------------------------------------
0          done, result on stdout         use it
1          agent failed (tool or model)   alert; read the JSON error
2          usage error (bad flags)        fix the invocation; do not retry
3          approval required              page a human; resume later with the thread id
4          budget exhausted               raise the budget or narrow the task
75         temporary failure (EX_TEMPFAIL) retry with backoff
130        interrupted (128 + SIGINT)     treat as cancelled
```

Code 2 matches what `argparse` already does on bad flags. Code 75 comes from the BSD `sysexits` convention for "try again later." Code 130 is the shell's convention for a process ended by Ctrl-C. Codes 3 and 4 are this tool's own; document them in `--help`.

```{code-cell} python
import argparse
import io
from langchain_core.messages import HumanMessage, ToolMessage

EXIT = {"ok": 0, "failed": 1, "usage": 2, "approval_required": 3, "budget": 4, "tempfail": 75, "interrupted": 130}


def build_parser():
    p = argparse.ArgumentParser(prog="supportctl", exit_on_error=False)
    p.add_argument("task", help="task text, or '-' to read it from stdin")
    p.add_argument("--output-format", choices=["text", "json", "stream-json"], default="text")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--non-interactive", action="store_true")
    return p


def main(argv, stdin, stdout, stderr, isatty=False, model_factory=acme_script, approver=None):
    try:
        args = build_parser().parse_args(argv)
    except (argparse.ArgumentError, SystemExit) as exc:
        stderr.write(f"usage error: {exc}\n")
        return EXIT["usage"]

    task = stdin.read().strip() if args.task == "-" else args.task
    interactive = isatty and not args.non_interactive     # no TTY means nobody can answer
    ev = Events(args.output_format, stdout, stderr)
    model, policy = model_factory(), PermissionPolicy()
    messages, pending, status = [HumanMessage(task)], [], "failed"
    ev.emit("start", task=task, interactive=interactive)

    try:
        for step in range(args.max_steps):
            ai = model.invoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                status = "approval_required" if pending else "ok"
                break
            for call in ai.tool_calls:
                decision = policy.decide(call["name"], interactive, approver)
                ev.emit("tool_call", step=step, name=call["name"], args=call["args"], decision=decision)
                if decision == "allow":
                    out = TOOLS[call["name"]].invoke(call["args"])
                elif decision == "pending":
                    pending.append({"tool": call["name"], "args": call["args"]})
                    out = "NOT EXECUTED: requires human approval; report it as pending."
                else:
                    out = "NOT EXECUTED: permission denied by policy."
                messages.append(ToolMessage(content=str(out), tool_call_id=call["id"]))
        else:
            status = "budget"
    except KeyboardInterrupt:
        status = "interrupted"

    result = {"status": status, "exit_code": EXIT[status], "answer": messages[-1].content,
              "pending_actions": pending, "steps": sum(isinstance(m, AIMessage) for m in messages)}
    if args.output_format == "text":
        stdout.write(result["answer"] + "\n")
    elif args.output_format == "json":
        stdout.write(json.dumps(result) + "\n")
    else:
        ev.emit("result", **result)         # always exactly one terminal event
    return result["exit_code"]


# 1. cron: headless, JSON. stdout must parse; progress must be on stderr.
REFUNDS_ISSUED.clear()
out, err = io.StringIO(), io.StringIO()
code = main(["Why is Acme upset?", "--output-format", "json"], io.StringIO(), out, err)
payload = json.loads(out.getvalue())                       # would raise if progress leaked
print("cron exit:", code, "| pending:", payload["pending_actions"])
assert code == 3 and REFUNDS_ISSUED == []                  # stopped, did not hang, did not refund
assert "[tool_call]" in err.getvalue()

# 2. another agent: stream-json, task on stdin. Every stdout line is one JSON event.
out, err = io.StringIO(), io.StringIO()
main(["-", "--output-format", "stream-json"], io.StringIO("Why is Acme upset?\n"), out, err)
events = [json.loads(line) for line in out.getvalue().splitlines()]
print("event types:", [e["type"] for e in events])
assert events[0]["type"] == "start" and events[-1]["type"] == "result"
assert err.getvalue() == ""                                # stream mode keeps stderr for real errors

# 3. on-call human at a TTY who approves.
out, err = io.StringIO(), io.StringIO()
code = main(["Why is Acme upset?"], io.StringIO(), out, err, isatty=True, approver=lambda name: True)
print("interactive exit:", code, "| refunds:", REFUNDS_ISSUED)
assert code == 0 and REFUNDS_ISSUED == [("in_881", 4800)]

# 4. usage error and budget exhaustion get their own codes.
assert main(["x", "--output-format", "yaml"], io.StringIO(), io.StringIO(), io.StringIO()) == 2
assert main(["x", "--max-steps", "1"], io.StringIO(), io.StringIO(), io.StringIO()) == 4
```

Look at what run 1 did. The agent read the CRM and the invoice, asked for a refund, was told "pending," wrote an answer, and the CLI exited 3 with the pending action in JSON. A cron wrapper can now page a human with exactly what needs approving.

But there is a hole. When the human says yes an hour later, what do we run? The from-scratch CLI kept its transcript in a Python list, and the list died with the process. The only option is to rerun the whole investigation, pay for it again, and hope the model proposes the same refund. That is the gap LangGraph's checkpointer fills.

## How LangGraph does it

The CLI contract stays. The loop inside it becomes a graph compiled with a checkpointer. Three primitives do the work:

```text
from-scratch CLI                          LangGraph CLI
-------------------------------------     --------------------------------------------
messages list in memory                   state saved per super-step under a thread_id
"pending" string fed back to the model    interrupt(): pause the graph at the tool node
rerun everything after approval           Command(resume=decision) continues the thread
events emitted by hand inside the loop    graph.stream(stream_mode="updates") -> events
```

The flow across two separate CLI invocations:

```text
run 1 (cron, headless)                          run 2 (human, later)
supportctl investigate acme                     supportctl --resume t-acme --decision approve
  agent ─► tools(read) ─► agent ─► tools          load checkpoint for t-acme
                                    │              tools node re-runs from the top
                              interrupt()          interrupt() returns "approve"
                                    │              stripe_refund executes
  checkpoint saved, exit 3 ◄────────┘              agent ─► final answer, exit 0
  stdout: {"status":"approval_required",
           "thread_id":"t-acme", ...}
```

```{code-cell} python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

CHECKPOINTS = InMemorySaver()      # a real CLI uses a file-backed saver so state survives the process
_model = {"m": None}               # one scripted model shared across both invocations


def agent_node(state: MessagesState):
    return {"messages": [_model["m"].invoke(state["messages"])]}


def tools_node(state: MessagesState):
    """Our own ToolNode with a permission gate. On resume this node re-runs from the top,
    so everything before interrupt() must be safe to repeat (here: pure policy checks)."""
    policy, results = PermissionPolicy(), []
    for call in state["messages"][-1].tool_calls:
        decision = policy.decide(call["name"], interactive=False)
        if decision == "pending":
            decision = interrupt({"tool": call["name"], "args": call["args"]})   # pause here
            decision = "allow" if decision == "approve" else "deny"
        out = TOOLS[call["name"]].invoke(call["args"]) if decision == "allow" else "NOT EXECUTED: denied"
        results.append(ToolMessage(content=str(out), tool_call_id=call["id"]))
    return {"messages": results}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END, ["tools", END])
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=CHECKPOINTS)


def graph_main(argv, stdin, stdout, stderr):
    p = argparse.ArgumentParser(prog="supportctl", exit_on_error=False)
    p.add_argument("task", nargs="?")
    p.add_argument("--thread-id", default="t-acme")
    p.add_argument("--resume", metavar="THREAD_ID")
    p.add_argument("--decision", choices=["approve", "reject"])
    p.add_argument("--max-steps", type=int, default=12)
    args = p.parse_args(argv)

    thread = args.resume or args.thread_id
    config = {"configurable": {"thread_id": thread}, "recursion_limit": args.max_steps}
    graph_input = Command(resume=args.decision) if args.resume else {"messages": [HumanMessage(args.task)]}

    for update in graph.stream(graph_input, config, stream_mode="updates"):
        for node, data in update.items():
            if node == "__interrupt__":
                stdout.write(json.dumps({"type": "approval_required", "thread_id": thread,
                                         "request": data[0].value,
                                         "resume": f"supportctl --resume {thread} --decision approve"}) + "\n")
                return EXIT["approval_required"]
            stdout.write(json.dumps({"type": "node", "node": node}) + "\n")
    answer = graph.get_state(config).values["messages"][-1].content
    stdout.write(json.dumps({"type": "result", "status": "ok", "thread_id": thread, "answer": answer}) + "\n")
    return EXIT["ok"]


REFUNDS_ISSUED.clear()
_model["m"] = acme_script()

out1 = io.StringIO()
code1 = graph_main(["Why is Acme upset?"], io.StringIO(), out1, io.StringIO())
last1 = json.loads(out1.getvalue().splitlines()[-1])
print("run 1 exit:", code1, "->", last1["resume"])
assert code1 == 3 and REFUNDS_ISSUED == []                       # paused before the side effect

out2 = io.StringIO()
code2 = graph_main(["--resume", "t-acme", "--decision", "approve"], io.StringIO(), out2, io.StringIO())
last2 = json.loads(out2.getvalue().splitlines()[-1])
print("run 2 exit:", code2, "->", last2["answer"])
assert code2 == 0 and REFUNDS_ISSUED == [("in_881", 4800)]       # executed exactly once, after approval
# the model was invoked 3 times total across both runs: the investigation was not repeated
assert sum(isinstance(m, AIMessage) for m in graph.get_state({"configurable": {"thread_id": "t-acme"}}).values["messages"]) == 3
```

The last assert is the payoff. Three model calls across two processes, the same as one uninterrupted run. The investigation was not repeated, and the refund ran exactly once, after a human said yes.

Map the pieces back:

| From-scratch CLI | LangGraph CLI | What happens underneath |
|---|---|---|
| `for step in range(max_steps)` | `recursion_limit` in config | The runtime counts super-steps and raises `GraphRecursionError` past the limit; map it to exit 4 |
| `ev.emit("tool_call", ...)` inside the loop | `graph.stream(..., stream_mode="updates")` | After each super-step the runtime yields `{node: update}`; the CLI translates it to NDJSON |
| `"pending"` fed back as text | `interrupt(payload)` | The runtime saves the checkpoint and surfaces `__interrupt__` in the stream |
| rerun everything | `Command(resume=value)` with the same `thread_id` | Loads the checkpoint, re-enters the interrupted node from the top, and `interrupt()` returns `value` |
| `messages` list | `MessagesState` + checkpointer | Every super-step's state is written under `thread_id`, so the process can exit between steps |

Two production details. First, `InMemorySaver` lives only as long as the Python process; the two "invocations" above share one process to keep the cell offline. A real CLI uses a file- or database-backed saver (SQLite for a laptop tool, Postgres for a shared service) so run 2 can be a new process. Second, the tools node re-runs from the top on resume. Our node does only policy checks before `interrupt()`, which are safe to repeat. If it had executed a read-only lookup first, that is fine; if it had executed a write first, that write would happen twice.

### Claude Code as the reference design

Claude Code's headless mode maps onto every decision above. Current docs describe these flags (check `claude --help` on your version before scripting against them):

```text
claude -p "summarize failing tests"            print mode: run once, no interactive UI
  --output-format text | json | stream-json    result shape; stream-json is one JSON event per line
  --allowedTools "Bash(npm test:*)" "Read"     allow rules, pattern-scoped per tool
  --disallowedTools "Bash(rm:*)"               deny rules
  --permission-mode default | acceptEdits | plan | bypassPermissions
  --max-turns 10                               step budget
  --resume <session-id> / --continue           resume a saved session
```

The mapping to this lesson:

| Decision | Our `supportctl` | Claude Code |
|---|---|---|
| Headless switch | `--non-interactive` or no TTY | `-p` print mode |
| Output modes | `text`, `json`, `stream-json` | the same three names |
| Permissions | allow / deny / ask with globs | allow / deny rules, permission modes, and hooks that can veto a tool call |
| Budget | `--max-steps` | `--max-turns` |
| Resume | `--resume THREAD_ID` | `--resume SESSION_ID` |
| Durable state | checkpointer | session transcript on disk |

The live cell runs our CLI against a real model. The only change is the model factory.

```{code-cell} python live
import os
from langchain.chat_models import init_chat_model

real = init_chat_model(os.environ.get("COURSE_MODEL", "openai:gpt-5.6")).bind_tools(list(TOOLS.values()))
REFUNDS_ISSUED.clear()
out, err = io.StringIO(), io.StringIO()
code = main(["Acme says their invoice doubled. Investigate and fix it.", "--output-format", "json"],
            io.StringIO(), out, err, model_factory=lambda: real)
print("exit", code)
print(json.dumps(json.loads(out.getvalue()), indent=2))
print("refunds actually issued:", REFUNDS_ISSUED)   # expect [] in headless mode
```

## What goes wrong in production

1. **Progress on stdout.** Symptom: `supportctl ... | jq` fails intermittently, only when a tool is slow enough to log a retry. Cause: one `print` for progress in a code path nobody tested with a pipe. Fix: route every human-facing line through one emitter that writes to stderr; test that stdout parses in every output mode, including failure paths.

2. **Hidden prompts in headless runs.** Symptom: CI jobs hang to timeout. Cause: an `input()` in a rarely hit branch, or a library that prompts for credentials. Fix: detect no-TTY and treat it as non-interactive; close stdin in CI; make "ask" resolve to "pending" with exit 3; add a wall-clock timeout that exits 75 or 1.

3. **Exit 0 on failure.** Symptom: the orchestrator marks a run green, and the answer says "I could not access Stripe." Cause: the loop finished normally, so the CLI returned 0. Fix: exit code reflects outcome, not whether the loop returned. Tool errors that the model could not recover from, pending actions, and budget exhaustion all get non-zero codes. For agent tasks, add an outcome check (did the expected artifact appear?) before exiting 0.

4. **Retry storms on non-idempotent runs.** Symptom: the scheduler retried on exit 1 and Acme got two refunds. Cause: "failed" and "temporary failure" shared a code, and the side effect had already happened. Fix: reserve 75 for "safe to retry," give side-effecting tools idempotency keys (a hash of thread id and tool-call id), and make resume, not rerun, the recovery path.

5. **Permission mode drift.** Symptom: a developer's convenience flag (`--yes`, or a bypass permission mode) ends up in a cron job and the agent runs destructive commands unattended. Cause: permissions were a flag, not a policy checked into the repo. Fix: keep deny rules in versioned config that flags cannot override; refuse bypass modes when not in a sandbox; log the effective policy at start.

6. **Unstructured errors.** Symptom: the parent agent receives `Traceback (most recent call last)` on stdout and hallucinates a cause. Cause: uncaught exceptions print tracebacks, and there was no terminal event. Fix: always emit exactly one terminal `result` event with `status`, `exit_code`, and a short error string; keep the traceback on stderr.

7. **Secrets in logs.** Symptom: API keys in CI logs. Cause: the CLI echoed its config or the tool arguments verbatim at debug level. Fix: redact known secret patterns in the emitter; pass credentials by environment or credential helper, never by flags (flags show up in `ps` and shell history).

## What changed by 2026

- **CLI agents became mainstream developer tools.** Claude Code, Codex CLI, Gemini CLI, and others made the terminal a primary surface for agents. Their headless modes (a print or exec mode, JSON and streaming-JSON output, allow/deny tool rules, turn limits, resumable sessions) converged on the contract this lesson builds.
- **Permissions moved from prompts to policy.** Allow and deny rules scoped by tool and argument pattern, plus hooks that run before a tool call, replaced "please be careful" instructions. Anthropic's managed-agents post describes keeping credentials out of the sandbox entirely, so a compromised run cannot exfiltrate them.
- **Stream-JSON became the agent-to-agent wire format on a single machine.** One JSON event per line is how an IDE, a CI system, or a parent agent consumes a CLI agent incrementally. It is the local cousin of the event streams in hosted agent APIs.
- **Resumable sessions are expected.** Long agent runs are resumed by session or thread id rather than restarted, which is the checkpointer pattern from 36. Durable Agent Orchestration applied to a CLI.
- **Instructions live in repo files.** Project instruction files that the CLI loads at start (CLAUDE.md, AGENTS.md) and skill folders loaded on demand (41. Skills) replaced long system-prompt flags. The CLI's job is to find and load them deterministically.
- **Sandboxing is the default for unattended runs.** Headless agents increasingly run inside containers or remote sandboxes with network egress limits (33. Sandboxed Agent Execution), so the permission policy is a second layer, not the only one.

## Interview questions

**Q1. You are shipping an agent as a CLI that will run both interactively and in CI. What is the contract?**

Strong: Results on stdout, everything for humans on stderr, and an exit code that tells the caller what happened. I would support three output formats: text for humans, a single JSON object for scripts, and newline-delimited JSON events for anything that wants progress. Headless mode is detected from the absence of a TTY or set by a flag, and in that mode the CLI never prompts. Permission questions resolve to "pending" with a distinct exit code and a resume command. State is checkpointed per thread so a later invocation can resume instead of restarting.

Weak: "Add a `--json` flag."

**Q2. The agent needs approval to issue a refund, but it is running from cron at 2 a.m. What happens?**

Strong: The run must not block and must not auto-approve. The tool call hits an "ask" rule, so the graph interrupts before the side effect, the checkpoint is saved under the thread id, and the CLI exits with a dedicated "approval required" code and a JSON payload describing the pending action and how to resume. The scheduler pages or tickets a human. When they approve, a new invocation resumes the same thread with the decision, the refund runs once, and the model finishes. The investigation is not repeated and the refund carries an idempotency key in case the resume itself is retried.

Weak: "It waits for the user to type yes."

**Q3. Which exit codes would you define, and why not just 0 and 1?**

Strong: The caller's reaction differs by cause, so the code must too. Usage errors (2) should never be retried. Temporary failures like rate limits (75, the sysexits "try again" code) should be retried with backoff. Approval required should page a human. Budget exhausted should raise a limit or narrow the task. Interruption (130) means cancelled. Collapsing these into 1 forces the caller to parse prose, and it causes the worst failure: retrying a non-idempotent run that already had side effects.

Weak: "0 is success, anything else is failure."

**Q4. How do you stream progress from a CLI agent without breaking scripts that parse its output?**

Strong: Either keep progress on stderr and put only the final result on stdout, or make stdout entirely machine-readable with one JSON event per line and a guaranteed terminal result event. Never mix. In LangGraph, `stream_mode="updates"` yields one update per node per super-step, and the CLI translates each into an event; interrupts arrive as a special `__interrupt__` update. The test is simple: in every mode and every failure path, stdout must parse.

Weak: "Print status messages as it goes."

**Q5. How should a CLI agent handle permissions for tools it has never seen, for example tools from a new MCP server?**

Strong: Fail closed. Unknown tools are denied until a rule allows them. Deny rules win over allow rules so a broad pattern cannot unlock something dangerous, and rules are scoped by arguments where possible, like allowing a test command but not arbitrary shell. The effective policy is logged at startup, lives in versioned config rather than only in flags, and bypass modes are only allowed inside a sandbox. That way adding a server adds capability only after someone writes a rule.

Weak: "Ask the user every time."

**Q6. Walk me through how Claude Code runs headless.**

Strong: Print mode runs one task without the interactive UI and exits. Output can be text, a single JSON result, or streaming JSON events. Tool access is controlled by allow and deny rules, which can be scoped to specific commands, plus a permission mode, and hooks can inspect or veto tool calls. A turn limit bounds the loop, and sessions can be resumed by id. Architecturally it is the same agent loop, with the Unix contract, a permission layer, and durable sessions around it. I would verify exact flag names on the installed version before scripting against them.

Weak: "You just run `claude` with a prompt."

**Q7. A parent agent calls your CLI agent as a tool. What does it need from you?**

Strong: A stable, documented schema for the final result, including status, exit code, answer, pending actions, and an error string, and a guarantee that exactly one terminal event is emitted even on failure. Output should be concise so it does not flood the parent's context; large artifacts go to files with paths in the result. The parent also needs bounded runtime (a timeout and a step limit) and idempotent behavior on retry. Essentially the CLI is a tool, and the tool-design rules apply: narrow purpose, structured result, actionable errors.

Weak: "It can read the stdout."

## Build it yourself

Extend `graph_main` into a complete CLI.

1. Add `--output-format text|json|stream-json` to `graph_main`, reusing the `Events` class, so it matches `main`.
2. Map `GraphRecursionError` (from `langgraph.errors`) to exit code 4 with a terminal `result` event whose status is `"budget"`.
3. Implement `--decision reject`: the refund is not executed, the model is told why, and the CLI exits 0 with `pending_actions == []` and the refund absent from `REFUNDS_ISSUED`.
4. Add a redacting emitter: any string matching `sk-[A-Za-z0-9]+` in an event is replaced by `sk-***` on both stdout and stderr.

Acceptance checks:

- Every line on stdout in `stream-json` mode parses with `json.loads`, and the last one has `type == "result"`, for success, approval-required, rejected, and budget runs.
- `--max-steps 2` exits 4 without calling `stripe_refund`.
- Reject path: `REFUNDS_ISSUED == []` and exit code 0.
- A tool argument containing `sk-live123` never appears verbatim in stdout or stderr.

## Key takeaways

- A CLI agent is the agent loop plus the Unix contract: stdout for results, stderr for humans, exit codes for callers.
- Offer text, JSON, and newline-delimited JSON events, and make stdout parse in every mode and every failure path.
- Headless mode never prompts; "ask" becomes a pending action with its own exit code.
- Distinct exit codes let callers react correctly: fix, retry, page a human, raise a budget, or stop.
- Permissions are policy, not flags: deny wins, unknown tools fail closed, rules live in versioned config.
- LangGraph's checkpointer and `interrupt` turn "needs approval" into a resumable pause across processes, so the investigation is not paid for twice.
- Claude Code's headless mode is a working reference for all of these decisions.
