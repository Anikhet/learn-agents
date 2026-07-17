require "json"

ROOT = File.expand_path("..", __dir__)
HEADING = "## Alternative framework implementation"

def source(cell)
  value = cell["source"]
  value.is_a?(Array) ? value.join : value.to_s
end

def markdown(text)
  {"cell_type" => "markdown", "metadata" => {}, "source" => text}
end

def code(text)
  {"cell_type" => "code", "execution_count" => nil, "metadata" => {}, "outputs" => [], "source" => text}
end

def inject(path, explanation, example)
  full_path = File.join(ROOT, path)
  notebook = JSON.parse(File.read(full_path))
  notebook["cells"].reject! do |cell|
    text = source(cell)
    (cell["cell_type"] == "markdown" && text.start_with?(HEADING)) ||
      (cell["cell_type"] == "code" && text.include?("# ALTERNATIVE FRAMEWORK EXAMPLE"))
  end
  first_code = notebook["cells"].index { |cell| cell["cell_type"] == "code" } || notebook["cells"].length
  notebook["cells"].insert(
    first_code,
    markdown("#{HEADING}\n\n#{explanation}\n"),
    code(example),
  )
  File.write(full_path, JSON.pretty_generate(notebook) + "\n")
end

inject(
  "notebooks/02_tools_and_protocols/05_tool_use_function_calling.ipynb",
  "OpenAI Agents SDK is useful here because `Runner` owns the complete tool loop. In Jupyter use the async API; `run_sync()` is documented as incompatible with an already-running event loop.",
  <<~'PY',
# ALTERNATIVE FRAMEWORK EXAMPLE: OpenAI Agents SDK
import os

if not os.getenv("OPENAI_API_KEY"):
    print("Skipped: set OPENAI_API_KEY to run this example.")
else:
    # Imports stay inside the enabled branch, so the base course does not require
    # optional packages. Install them with: uv sync --extra alternatives
    from agents import Agent, Runner, function_tool

    @function_tool
    def lookup_order(order_id: str) -> str:
        """Return the current status for one order ID."""
        return f"Order {order_id} is packed and ready to ship."

    agent = Agent(
        name="Order support",
        model="gpt-5.4",
        instructions="Use lookup_order for order status; never invent a status.",
        tools=[lookup_order],
    )

    # Jupyter supports top-level await. Runner executes tool calls and sends their
    # results back to the model until it produces a final answer.
    result = await Runner.run(agent, "Where is order A-104?")
    print(result.final_output)
PY
)

inject(
  "notebooks/02_tools_and_protocols/07_multi_agent_collaboration.ipynb",
  "OpenAI handoffs demonstrate transfer of conversation control. This differs from an agent-as-tool: after a handoff, the specialist becomes the active agent.",
  <<~'PY',
# ALTERNATIVE FRAMEWORK EXAMPLE: OpenAI Agents SDK handoffs
import os

if not os.getenv("OPENAI_API_KEY"):
    print("Skipped: set OPENAI_API_KEY to run this example.")
else:
    from agents import Agent, Runner

    billing_agent = Agent(
        name="Billing specialist",
        handoff_description="Handles invoices, refunds, and payment questions.",
        instructions="Resolve billing questions and explain the next action.",
        model="gpt-5.4",
    )
    technical_agent = Agent(
        name="Technical specialist",
        handoff_description="Handles errors, crashes, and troubleshooting.",
        instructions="Diagnose technical issues one safe step at a time.",
        model="gpt-5.4",
    )
    triage_agent = Agent(
        name="Triage",
        instructions="Answer simple questions or hand off to the correct specialist.",
        handoffs=[billing_agent, technical_agent],
        model="gpt-5.4",
    )

    result = await Runner.run(triage_agent, "The app crashes when I open an invoice.")
    print("Final agent:", result.last_agent.name)
    print(result.final_output)
PY
)

inject(
  "notebooks/06_production/24_structured_output_model_agnostic.ipynb",
  "Pydantic AI is included because typed outputs and validation retries are central abstractions rather than add-ons. The returned `result.output` is a validated Python object.",
  <<~'PY',
# ALTERNATIVE FRAMEWORK EXAMPLE: Pydantic AI validated output
import os
from pydantic import BaseModel, Field

class SupportDecision(BaseModel):
    category: str = Field(description="billing, technical, or general")
    urgency: int = Field(ge=1, le=5)
    needs_human: bool

if not os.getenv("OPENAI_API_KEY"):
    print("Skipped: set OPENAI_API_KEY to run this example.")
else:
    from pydantic_ai import Agent

    agent = Agent(
        "openai:gpt-5.4",
        output_type=SupportDecision,
        instructions="Classify the request. Escalate security or payment-risk cases.",
    )
    result = await agent.run("I was charged twice and do not recognize one payment.")

    # Pydantic has already checked the field names, types, and urgency range.
    decision = result.output
    print(decision.model_dump())
PY
)

inject(
  "notebooks/03_memory_rag_knowledge/14_knowledge_retrieval_rag.ipynb",
  "LlamaIndex is included because ingestion, indexing, retrieval, and query engines are its primary abstraction. This example uses a tiny in-memory corpus so each stage stays visible.",
  <<~'PY',
# ALTERNATIVE FRAMEWORK EXAMPLE: LlamaIndex RAG
import os

if not os.getenv("OPENAI_API_KEY"):
    print("Skipped: set OPENAI_API_KEY to build embeddings and answer the query.")
else:
    from llama_index.core import Document, Settings, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    # Documents are the source records. Metadata travels with retrieved chunks.
    documents = [
        Document(text="Refunds are available for 30 days after purchase.", metadata={"source": "refund-policy"}),
        Document(text="Enterprise plans include SSO and audit logs.", metadata={"source": "enterprise-plan"}),
    ]

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-5.4")

    # The index embeds documents; the query engine retrieves relevant chunks and
    # asks the model to answer from them.
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=2)
    response = query_engine.query("How long do customers have to request a refund?")

    print(response)
    print([node.metadata["source"] for node in response.source_nodes])
PY
)

inject(
  "notebooks/06_production/28_production_agent_harness.ipynb",
  "Agno is included here because its `Agent`, `Team`, and deterministic `Workflow` objects can be served and managed through AgentOS. This example shows the SDK object that later becomes a production service.",
  <<~'PY',
# ALTERNATIVE FRAMEWORK EXAMPLE: Agno agent prepared for AgentOS
import os

if not os.getenv("OPENAI_API_KEY"):
    print("Skipped: set OPENAI_API_KEY to run this Agno example.")
else:
    from agno.agent import Agent

    def lookup_runbook(service: str) -> str:
        """Return the first recovery step for a known service."""
        runbooks = {"checkout": "Check payment-provider health and queue depth."}
        return runbooks.get(service, "Escalate to the owning team.")

    agent = Agent(
        id="operations-assistant",
        name="Operations assistant",
        model="openai:gpt-5.4",
        instructions="Use the runbook tool. State evidence before recommending action.",
        tools=[lookup_runbook],
        markdown=True,
    )

    # Agno runs the same Agent object directly during development; AgentOS can
    # later expose it through managed run/session APIs.
    agent.print_response("Checkout errors increased. What should I inspect first?")
PY
)

puts "Added selective alternative-framework examples."
