# 23. Voice AI Fundamentals: the agent loop under a one-second deadline

*Track A (agent engineering). Book: course addition with no book chapter; it applies Chapter 5 (Tool Use) and Chapter 13 (Human in the Loop) under real-time constraints. Prerequisites: 00. What an agent actually is, 05. Tool Use, 38. LLM Serving Infrastructure.*

A voice agent is the same loop you built in lesson 00 with two things changed: the input and output are audio, and the whole turn has to finish in about the time it takes a person to draw a breath. Everything hard about voice follows from that deadline. You cannot wait for the model to finish before you start speaking. You cannot wait for the user to finish before you start listening. You cannot make a four-second tool call in silence. And when the user interrupts, your transcript has to reflect what they actually heard, not what you planned to say.

This lesson builds the timing machinery: the latency budget, the turn-taking state machine, barge-in, and tool use that does not leave dead air. The model is scripted so the control flow is the object of study.

## The problem

The support agent from this course gets a phone line. A customer calls:

> "Hi, I was charged twice for the Acme workspace last month."

The naive design is the text agent with a microphone bolted on:

```text
record until silence ──► STT (whole utterance) ──► LLM (whole answer) ──► TTS (whole answer) ──► play
     ~1.0 s wait           ~0.6 s                     ~2.5 s                  ~1.2 s            
                                                                              total: ~5.3 s of silence
```

Five seconds of silence on a phone call is an eternity. Humans respond to each other in roughly 200 to 500 ms; past about a second, callers assume the line dropped and say "hello?", which the system hears as new input, which restarts the pipeline. Meanwhile the agent looks up the Stripe charge, which takes three seconds, during which nothing is said.

The fix is not a faster model. It is a different shape:

```text
audio in ──► VAD ──► streaming STT ──┐ partial transcripts
                                     ▼
                              endpointing: "has the user finished?"
                                     ▼
                              LLM (streaming tokens) ──► sentence splitter ──► streaming TTS ──► play
                                     ▲                                            │
                              tool calls (with a filler                    barge-in: user speaks,
                              "let me check that" while waiting)           stop playback, truncate history
```

Every stage streams into the next, so work overlaps instead of queueing. The budget that matters is **voice-to-voice latency**: from the moment the user stops speaking to the moment the agent's first audio reaches them. Practitioners aim for well under a second at p95; the components are roughly:

| Stage | Typical contribution | What controls it |
|---|---|---|
| endpointing wait | 200-600 ms | VAD silence threshold, semantic end-of-turn model |
| STT final transcript | 100-300 ms | streaming recognizer, model size |
| LLM time-to-first-token | 200-800 ms | prompt length, prefix cache hits, model tier, queueing |
| TTS time-to-first-byte | 80-250 ms | streaming synthesizer, voice model |
| network and jitter buffer | 40-150 ms | WebRTC vs telephony, region |

Add the typical values and you are already at about a second. There is no slack, which is why every design decision in voice is a latency decision first.

### Cascaded pipeline versus speech-to-speech

There are two architectures:

```text
Cascaded (STT -> LLM -> TTS)                Speech-to-speech (one multimodal model)
  + pick best-in-class per stage               + lowest latency, hears tone and prosody
  + full transcript, easy tool use, evals      + natural interruptions, no sentence splitter
  + swap any stage independently               - transcript is a by-product, may not match audio
  - three latency budgets to sum               - tool use, guardrails, evals harder to bolt on
  - loses prosody, sarcasm, hesitation         - fewer vendor choices, harder to swap
```

The course's position: build the cascaded pipeline first because you can see every stage and test every stage. Move to speech-to-speech when latency or naturalness is the product, and keep a text transcript alongside it for evals and audit. Most production systems in 2026 are hybrids: a speech-to-speech model for the conversational surface with a text agent behind it for tools and policy.

## Build it without a framework

Two things need to be built by hand before any framework will make sense: the budget arithmetic and the turn-taking state machine.

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

### The latency budget, sequential versus streamed

The difference between the naive and the streamed pipeline is not the sum of the stages. It is *which* parts of each stage sit on the critical path. In the streamed pipeline, only each stage's time-to-first-output counts; the rest overlaps with the next stage.

```{code-cell} python
import random
import statistics

random.seed(23)

# (name, time to first output, time to complete) in ms, sampled with jitter.
# The critical path of a streamed pipeline is the sum of first-output times;
# a sequential pipeline pays the completion time of every stage.
STAGES = {
    "endpoint_wait": (350, 350),       # silence threshold before we decide the turn ended
    "stt_final":     (150, 150),       # streaming recognizer has partials; final lands here
    "llm":           (450, 2400),      # TTFT vs full answer decode
    "tts":           (120, 1300),      # first audio byte vs whole utterance synthesized
    "network":       (60, 60),
}


def sample(ms):
    return max(0.0, random.gauss(ms, ms * 0.25))


def voice_to_voice(stages, streamed):
    total = 0.0
    for name, (first, full) in stages.items():
        total += sample(first if streamed else full)
    return total


def pct(xs, p):
    return sorted(xs)[int(p / 100 * len(xs)) - 1]


seq = [voice_to_voice(STAGES, streamed=False) for _ in range(2000)]
strm = [voice_to_voice(STAGES, streamed=True) for _ in range(2000)]
print(f"sequential : p50 {pct(seq, 50):5.0f} ms   p95 {pct(seq, 95):5.0f} ms")
print(f"streamed   : p50 {pct(strm, 50):5.0f} ms   p95 {pct(strm, 95):5.0f} ms")
assert pct(strm, 95) < 0.45 * pct(seq, 95)

# The budget view an interviewer wants: where does the second go, and what is the biggest lever?
first_outputs = {k: v[0] for k, v in STAGES.items()}
budget = sum(first_outputs.values())
for k, v in sorted(first_outputs.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<14} {v:4d} ms  {'#' * (v // 25)}")
print(f"  {'critical path':<14} {budget:4d} ms")
assert max(first_outputs, key=first_outputs.get) == "llm"

# Halving LLM TTFT (prefix caching, smaller model tier) beats halving TTS time-to-first-byte 4:1.
faster_llm = dict(STAGES, llm=(225, 2400))
faster_tts = dict(STAGES, tts=(60, 1300))
gain_llm = budget - sum(v[0] for v in faster_llm.values())
gain_tts = budget - sum(v[0] for v in faster_tts.values())
print(f"gain from halving LLM TTFT: {gain_llm} ms; from halving TTS TTFB: {gain_tts} ms")
assert gain_llm > 3 * gain_tts
```

Two things to notice. First, streaming does not make any stage faster; it removes completion times from the critical path, which is the same TTFT-versus-total argument as in 38. LLM Serving Infrastructure. Second, the biggest lever is the LLM's time-to-first-token, and in a voice agent that is dominated by prompt length and prefix-cache hits. A voice system prompt that changes on every turn is a latency bug.

### Turn-taking: VAD, endpointing, and barge-in

Voice activity detection (VAD) answers a narrow question every 20 or 30 ms: is there speech in this frame? Endpointing answers the harder one: has the user *finished their turn*? Silence alone is a poor signal. People pause mid-sentence ("I was charged... twice"), and a fixed 300 ms silence threshold cuts them off, while a 1,200 ms threshold makes every response feel sluggish. Modern endpointers combine silence duration with a semantic signal: does the transcript so far look complete?

Barge-in is the other half. While the agent speaks, the user may talk over it. The system must stop playback quickly, discard queued audio, and, critically, record in the transcript only what the user actually heard. Otherwise the agent's memory says it explained the refund policy when the user cut it off at "Let me".

```text
        speech detected                      end of turn (silence >= T and transcript complete)
LISTENING ────────────► USER_SPEAKING ─────────────────────────────────────► THINKING
    ▲                        ▲                                                   │
    │                        │ barge-in: user speech while we play              │ first sentence ready
    │                        │ (stop playback, truncate what was said)          ▼
    └──── playback done ─────┴──────────────────────────────────────────── AGENT_SPEAKING
```

```{code-cell} python
from dataclasses import dataclass, field


@dataclass
class TurnTaker:
    """Turn-taking state machine over 20 ms audio frames; the transcript is what was actually heard."""
    frame_ms: int = 20
    silence_ms_to_end: int = 400
    state: str = "LISTENING"
    silence_ms: int = 0
    transcript: list = field(default_factory=list)          # what the user heard / said, in order
    playback: list = field(default_factory=list)            # (word, start_ms) scheduled TTS words
    play_started_at: int = 0
    now_ms: int = 0
    events: list = field(default_factory=list)

    def on_frame(self, speech: bool, partial_text: str = "", looks_complete: bool = False):
        self.now_ms += self.frame_ms
        if self.state == "LISTENING" and speech:
            self.state = "USER_SPEAKING"
        elif self.state == "USER_SPEAKING":
            if speech:
                self.silence_ms = 0
            else:
                self.silence_ms += self.frame_ms
                # semantic endpointing: end early if the sentence looks finished,
                # otherwise wait for the full silence threshold
                threshold = self.silence_ms_to_end // 2 if looks_complete else self.silence_ms_to_end
                if self.silence_ms >= threshold:
                    self.state, self.silence_ms = "THINKING", 0
                    self.transcript.append(("user", partial_text))
                    self.events.append(("end_of_turn", self.now_ms))
        elif self.state == "AGENT_SPEAKING" and speech:
            self.barge_in()

    def start_speaking(self, words_with_offsets):
        self.state, self.playback, self.play_started_at = "AGENT_SPEAKING", words_with_offsets, self.now_ms

    def barge_in(self):
        elapsed = self.now_ms - self.play_started_at
        heard = [w for w, start in self.playback if start < elapsed]
        self.transcript.append(("agent", " ".join(heard) + (" [interrupted]" if heard else "[interrupted]")))
        self.events.append(("barge_in", self.now_ms, len(heard)))
        self.state, self.playback = "USER_SPEAKING", []

    def finish_speaking(self):
        self.transcript.append(("agent", " ".join(w for w, _ in self.playback)))
        self.state, self.playback = "LISTENING", []


tt = TurnTaker()
# User speaks 600 ms, pauses 200 ms mid-sentence (must NOT end the turn), speaks 400 ms, stops.
for _ in range(30):
    tt.on_frame(True, "I was charged")
for _ in range(10):
    tt.on_frame(False, "I was charged", looks_complete=False)
assert tt.state == "USER_SPEAKING", "a mid-sentence pause must not end the turn"
for _ in range(20):
    tt.on_frame(True, "I was charged twice last month")
for _ in range(30):
    tt.on_frame(False, "I was charged twice last month.", looks_complete=True)
assert tt.state == "THINKING" and tt.events[0][0] == "end_of_turn"
end_at = tt.events[0][1]
assert end_at - 1200 == 200, "semantic endpointing ended the turn after 200 ms, not 400"

# Agent starts a long answer; the user interrupts 1 second in.
tts_words = [("I", 0), ("see", 150), ("two", 300), ("charges", 500), ("on", 800), ("the", 900),
             ("Acme", 1000), ("workspace.", 1250), ("Let", 1600), ("me", 1750), ("check", 1900)]
tt.start_speaking(tts_words)
for _ in range(49):
    tt.on_frame(False)
tt.on_frame(True, "actually")                 # interrupt lands at 1000 ms, before "Acme" plays
assert tt.state == "USER_SPEAKING"
for who, text in tt.transcript:
    print(f"{who:>5}: {text}")
assert tt.transcript[-1] == ("agent", "I see two charges on the [interrupted]")
print("barge-in kept only the words the user actually heard")
```

The `looks_complete` flag is where a semantic endpointer plugs in. In production it is a small classifier over the partial transcript (and sometimes prosody), and it is the difference between a bot that talks over people and one that does not.

## How LangGraph does it

LangGraph does not do audio. What it gives a voice agent is the same thing it gives a text agent: explicit state, nodes that can be streamed, a checkpointer so a call survives a worker restart, and interrupts for approvals. The audio stages (VAD, STT, TTS) are I/O adapters around the graph; the graph owns the turn.

The one design constraint voice imposes on the graph is **no dead air during tools**. A Stripe lookup takes two to four seconds. The pattern is: when the model decides to call a tool, immediately speak a short filler ("Let me pull up that account"), run the tool, then speak the answer. In graph terms, the routing edge that sends control to the tool node also emits a spoken acknowledgment.

```{code-cell} python
from typing import Annotated
from operator import add
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver


@tool
def stripe_charges(customer: str) -> str:
    """List recent charges for a customer by account name."""
    return "2 charges on 2026-08-01: ch_1 $499.00, ch_2 $499.00 (duplicate, same idempotency window)"


class VoiceState(MessagesState):
    spoken: Annotated[list, add]          # sentences actually sent to TTS, in order
    tool_filler_used: bool


FILLERS = {"stripe_charges": "Let me pull up the charges on that account."}

model = scripted(
    [("stripe_charges", {"customer": "Acme"})],
    "I see two charges of $499 on August first. The second one is a duplicate. I can refund it now if you like.",
).bind_tools([stripe_charges])


def agent(state: VoiceState):
    return {"messages": [model.invoke(state["messages"])]}


def speak_filler(state: VoiceState):
    """Runs on the way to the tool so the caller never hears silence."""
    call = state["messages"][-1].tool_calls[0]
    return {"spoken": [FILLERS.get(call["name"], "One moment.")], "tool_filler_used": True}


def speak_answer(state: VoiceState):
    """Sentence-split the final answer; each sentence is a TTS request that can start streaming."""
    text = state["messages"][-1].content
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    return {"spoken": sentences}


def route(state: VoiceState):
    return "speak_filler" if state["messages"][-1].tool_calls else "speak_answer"


builder = StateGraph(VoiceState)
builder.add_node("agent", agent)
builder.add_node("speak_filler", speak_filler)
builder.add_node("tools", ToolNode([stripe_charges]))
builder.add_node("speak_answer", speak_answer)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route, ["speak_filler", "speak_answer"])
builder.add_edge("speak_filler", "tools")
builder.add_edge("tools", "agent")
builder.add_edge("speak_answer", END)
graph = builder.compile(checkpointer=InMemorySaver())     # a call is a thread; a restart resumes it

config = {"configurable": {"thread_id": "call-7781"}}
order = []
for event in graph.stream({"messages": [HumanMessage("I was charged twice for the Acme workspace.")],
                           "spoken": [], "tool_filler_used": False}, config, stream_mode="updates"):
    node = next(iter(event))
    order.append(node)
    if "spoken" in event[node]:
        for s in event[node]["spoken"]:
            print(f"  TTS <- {s}")

print("node order:", " -> ".join(order))
assert order == ["agent", "speak_filler", "tools", "agent", "speak_answer"]
final = graph.get_state(config).values
assert final["spoken"][0].startswith("Let me pull up")
assert len(final["spoken"]) == 4 and final["tool_filler_used"]
```

Map it to the from-scratch version. The `spoken` reducer with `operator.add` is the `transcript` list from `TurnTaker`: an append-only record of what was sent to the speaker. `stream_mode="updates"` is how the audio layer learns that a sentence is ready without waiting for the graph to finish; that is the streaming boundary between the graph and TTS. The checkpointer makes the call a durable thread: if the worker holding the WebSocket dies, a new worker can load `call-7781` and continue from the last node.

### Barge-in as a state correction

When the user interrupts mid-answer, the audio layer knows which words were played. The graph state should be corrected to match, and `update_state` is the tool for it: it writes a state patch as if a node had produced it.

```{code-cell} python
# The audio layer reports that playback stopped after the first two sentences.
heard = final["spoken"][:2]
truncated_answer = " ".join(heard[1:]) + " [interrupted]"          # heard[0] is the filler
graph.update_state(config, {"messages": [AIMessage(content=truncated_answer,
                                                    id=final["messages"][-1].id)]})
corrected = graph.get_state(config).values["messages"][-1].content
print("model memory after barge-in:", corrected)
assert "refund" not in corrected and corrected.endswith("[interrupted]")

# The next turn continues from the corrected transcript, with the user's interruption appended.
next_turn = graph.get_state(config).values["messages"] + [HumanMessage("Wait, which one is the duplicate?")]
assert sum(isinstance(m, ToolMessage) for m in next_turn) == 1
print(f"{len(next_turn)} messages carried into the next turn")
```

Passing the same message `id` makes `add_messages` *replace* the AI message rather than append a second one. That is the reducer doing what `TurnTaker.barge_in` did by hand.

### Where a voice framework earns its place

The OpenAI Agents SDK ships a voice pipeline that wraps exactly this structure: a streaming STT stage, your agent as the workflow, and a streaming TTS stage, with events for audio chunks and lifecycle. It is worth a look because it shows the seams a production pipeline needs even when the agent logic is tiny.

```{code-cell} python live
# Requires the optional openai-agents package with voice extras and an OpenAI key.
import os
import asyncio
from agents import Agent, function_tool
from agents.voice import SingleAgentVoiceWorkflow, VoicePipeline, AudioInput
import numpy as np


@function_tool
def stripe_charges(customer: str) -> str:
    """List recent charges for a customer by account name."""
    return "2 charges on 2026-08-01, the second is a duplicate."


agent = Agent(name="support", instructions="Answer in one or two short sentences.", tools=[stripe_charges])
pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))


async def demo():
    silence = np.zeros(24000 * 2, dtype=np.int16)          # 2 s of silence stands in for a mic
    result = await pipeline.run(AudioInput(buffer=silence))
    async for event in result.stream():                    # audio chunks and lifecycle events
        print(event.type)


asyncio.run(demo())
```

The SDK's current docs describe the pipeline as STT, then your workflow, then TTS, with lifecycle events you can use to drive the turn-taking machine above; the agent code inside is unchanged from a text agent, which is the point.

## What goes wrong in production

1. **The agent talks over the caller.** Symptom: users complain the bot "does not listen." Cause: a fixed silence threshold that is too short, no semantic endpointing, or barge-in detection that only triggers on loud speech. Fix: combine silence duration with a completeness signal; lower the threshold only when the transcript looks finished; test with recordings of hesitant speakers, not just clean scripts.

2. **Transcript drift after barge-in.** Symptom: the agent refers to things it "said" that the caller never heard, or repeats itself. Cause: the full planned answer was appended to history before playback finished. Fix: append only what was played, using word timestamps from TTS, and mark the interruption. This is a correctness bug, not a polish item.

3. **Dead air during tools.** Symptom: three seconds of silence, then "hello?", then two overlapping answers. Cause: the tool call blocks with nothing spoken. Fix: speak a filler on the way to the tool, keep tools under a hard timeout, and treat the caller's "hello?" during a tool call as a non-interrupting utterance.

4. **Voice-to-voice p95 creeps past a second.** Symptom: the demo was fast; production is sluggish. Cause: the system prompt grows, the prompt is rebuilt with volatile content so the prefix cache misses, or a bigger model tier was swapped in. Fix: budget TTFT per turn, keep the prefix stable (see 38. LLM Serving Infrastructure), and use a small, fast model for the conversational turn with a larger model behind a tool for hard reasoning.

5. **Numbers and names are mangled.** Symptom: "ch underscore one" and "four nine nine point zero zero dollars" read aloud; order ids misheard. Cause: text-optimized tool outputs and prompts sent straight to TTS; STT without a vocabulary boost. Fix: a spoken-form normalizer before TTS, a "say it for the ear" instruction in the voice system prompt, keyword boosting in STT for product names, and confirmation of critical identifiers ("that is order ending in 4-4-2, correct?").

6. **A worker restart drops the call's memory.** Symptom: after a deploy, callers mid-conversation are greeted as new. Cause: conversation state lived in process memory next to the WebSocket. Fix: state in a checkpointer keyed by call id; the media session and the conversation state are different lifetimes (27. Systems Design: Voice Platform).

7. **The guardrail is on the wrong path.** Symptom: an unsafe or off-policy sentence is spoken before the moderation check returns. Cause: guardrails were run on the full answer while TTS was already streaming sentence one. Fix: run fast checks per sentence before it enters the TTS queue; run the expensive checks in parallel and be willing to cut playback and correct.

## What changed by 2026

- **Speech-to-speech went from demo to option.** Realtime multimodal models that take audio in and produce audio out are now offered by the major providers and supported in the OpenAI Agents SDK's realtime and voice components. Current docs still describe the cascaded pipeline alongside them, and most teams run a hybrid: speech-to-speech for the surface, a text agent for tools and policy.
- **Endpointing became semantic.** Silence-threshold turn detection is now the fallback; current provider docs describe turn-detection models that use transcript content and prosody. Test them against your callers' speech patterns; they are trained on someone else's.
- **Tool use in voice matured into a pattern.** The filler-then-tool-then-answer shape, per-sentence guardrails, and spoken-form normalization are now standard components rather than custom work. What has not changed: tools must have hard timeouts, because silence is the one failure a caller cannot tolerate.
- **Durable sessions arrived.** Checkpointed conversation state (LangGraph threads, Agents SDK sessions) is expected, so a media-server restart no longer means a lost conversation.
- **Evals for voice split into layers.** Teams now evaluate STT word error rate on their own audio, turn-taking metrics (interruption rate, false end-of-turn rate), voice-to-voice latency percentiles, and task success separately. A transcript-only eval hides the first three.

## Interview questions

**Q1. What is the latency budget for a voice agent and where does it go?**

Strong: The metric is voice-to-voice latency, from the user's last word to the agent's first audio, and the target is well under a second at p95 because human turn gaps are a few hundred milliseconds. On a streamed cascaded pipeline the critical path is the sum of time-to-first-output for each stage: endpointing wait of 200 to 600 ms, STT finalization of 100 to 300 ms, LLM time-to-first-token of 200 to 800 ms, TTS time-to-first-byte of 80 to 250 ms, and network. The LLM's TTFT is the biggest and most controllable term: prompt length, prefix caching, and model tier. Streaming does not make stages faster; it removes their completion times from the critical path.

Weak: "Use fast models and a good TTS."

**Q2. Cascaded STT-LLM-TTS versus speech-to-speech: how do you choose?**

Strong: Cascaded gives me best-of-breed per stage, a real transcript, straightforward tool use, per-stage evals, and vendor swapability, at the cost of summing three latency budgets and losing prosody. Speech-to-speech gives lower latency and more natural interruption handling and hears tone, but the transcript is a by-product, tool use and guardrails are harder to insert, and I am tied to one vendor. I would start cascaded because it is testable, move the conversational surface to speech-to-speech when latency or naturalness is the product, and keep a text agent behind it for tools and policy.

Weak: "Speech-to-speech is newer so it is better."

**Q3. How does barge-in work and what is the subtle bug?**

Strong: While the agent plays audio, VAD keeps running on the input. When the user speaks, we stop playback immediately, flush the queued TTS audio, and switch the state machine to listening. The subtle bug is the transcript: if we already appended the full planned answer, the model now believes it said things the user never heard. Using word timestamps from TTS, we truncate the assistant message to what was actually played and mark it interrupted. In LangGraph that is `update_state` with the same message id so the reducer replaces rather than appends.

Weak: "Stop the audio when the user talks."

**Q4. How do you handle a three-second tool call in a voice conversation?**

Strong: Never in silence. When the model emits a tool call, the routing step also emits a short spoken filler appropriate to the tool, then runs the tool under a hard timeout of a couple of seconds, then speaks the answer. If the tool exceeds its timeout I say so and offer a callback or a follow-up rather than leaving dead air. Utterances like "hello?" during the wait are classified as non-interrupting so they do not restart the turn. For long operations I move them out of the call entirely and use the conversation to confirm and hand off.

Weak: "Make the tool faster."

**Q5. What is the difference between VAD and endpointing?**

Strong: VAD is a frame-level classifier: is there speech in this 20 ms window. Endpointing is the turn-level decision: has the user finished. Silence alone is a bad endpointer because people pause mid-thought, so a short threshold cuts them off and a long one makes the agent sluggish. Modern endpointers combine silence duration with a semantic completeness signal from the partial transcript, and sometimes prosody, so a complete-looking sentence ends the turn after a short pause while an incomplete one waits longer. I test this on recordings of hesitant speakers, not on clean scripts.

Weak: "VAD detects when the user stops talking."

**Q6. Where does conversation state live for a phone call, and why does it matter?**

Strong: Not in the process that holds the media socket. The media session and the conversation are different lifetimes: a WebSocket can reconnect or a worker can be redeployed while the conversation continues. I keep the transcript and agent state in a checkpointer keyed by call id, LangGraph threads or an equivalent session store, and I write after every node so a new worker can resume. The audio layer holds only the buffers it needs for the current turn.

Weak: "In memory, calls are short."

**Q7. How would you evaluate a voice agent?**

Strong: In layers, because a transcript-only eval hides most voice failures. STT word error rate on our own callers' audio, with attention to names and numbers. Turn-taking metrics: false end-of-turn rate, interruption latency, barge-in handling correctness. Latency: voice-to-voice p50 and p95 per turn, broken down by stage. Then task success and policy adherence on the transcript, the same way I would evaluate the text agent, using recorded calls replayed through the pipeline so regressions are caught before callers find them.

Weak: "Listen to some calls."

## Build it yourself

Extend the `TurnTaker` and the LangGraph voice graph:

1. Add a `tool_timeout_ms` to the graph: if the tool node exceeds it (simulate with a slow tool), route to a `speak_timeout` node that says "This is taking longer than expected; I will text you the result" and ends the turn.
2. Add a `hello_filter`: while state is `THINKING` or a tool is running, utterances shorter than 400 ms that match a small set ("hello", "you there") do not start a new turn.
3. Add a per-sentence guardrail node between `speak_answer` and the TTS queue that blocks any sentence containing a full card number pattern, replacing it with a spoken redaction.
4. Record per-turn voice-to-voice latency (simulated) in the state and print p50 and p95 over 100 simulated turns.

Acceptance checks:

- With a tool that sleeps past `tool_timeout_ms`, `spoken` ends with the timeout sentence and no answer sentence, and the graph reaches `END`.
- A 300 ms "hello" during `THINKING` leaves `state == "THINKING"` and adds nothing to the transcript.
- A sentence containing `4242 4242 4242 4242` never appears in `spoken`; the redacted version does.
- Over 100 turns, p95 latency in the state is below 1,000 ms when `llm` TTFT is 450 ms, and above it when TTFT is 900 ms.

## Key takeaways

- A voice agent is the text agent loop under a roughly one-second voice-to-voice deadline; every design choice is a latency choice first.
- Stream every stage so only time-to-first-output sits on the critical path; the LLM's TTFT is the biggest and most controllable term, and prefix stability is a latency feature.
- Endpointing is not VAD: combine silence with a semantic completeness signal or the agent will talk over people or feel sluggish.
- Barge-in must correct the transcript to what the user actually heard; in LangGraph that is a reducer replacing the message by id.
- Tools in voice follow filler, tool with hard timeout, answer; silence is the one failure a caller cannot tolerate.
- Conversation state outlives the media socket: checkpoint per node keyed by call id.
- Start cascaded because it is testable; adopt speech-to-speech for the surface when latency or naturalness is the product, and keep a text agent behind it for tools and policy.
