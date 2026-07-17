require "json"

ROOT = File.expand_path("..", __dir__)
MARKER = "### Read the next code cell"

PROMPT_CHAINING_CODE = <<~'PY'
# `os` is part of Python's standard library. We use it to read environment variables.
import os

# Import two LangChain building blocks:
# - `init_chat_model` creates a chat-model object.
# - `ChatPromptTemplate` creates reusable messages with placeholders.
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# API keys should be stored outside the notebook. `getenv` returns the key when
# present and returns `None` when it is missing.
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    # This branch runs when no key is configured, avoiding a failed paid request.
    print("Skipped: set OPENAI_API_KEY to run the LangChain chain.")
else:
    # Construct the model client. No API request happens on this line.
    # Temperature 0 asks for conservative sampling but does not guarantee correctness.
    model = init_chat_model("openai:gpt-5.4", temperature=0)

    # A prompt is a list of role/content tuples. `{request}` is a placeholder.
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract audience, task, and output format. Return concise JSON."),
        ("user", "{request}"),
    ])

    # LangChain overloads `|` to connect runnable components. The result accepts
    # a dictionary, formats the prompt, calls the model, and returns an AIMessage.
    extract_chain = extract_prompt | model

    draft_prompt = ChatPromptTemplate.from_messages([
        ("system", "Draft a useful answer from these requirements."),
        ("user", "{requirements}"),
    ])
    draft_chain = draft_prompt | model

    polish_prompt = ChatPromptTemplate.from_messages([
        ("system", "Make this concise and actionable without adding unsupported facts."),
        ("user", "{draft}"),
    ])
    polish_chain = polish_prompt | model

    # First API call: replace `{request}` and extract a requirements artifact.
    requirements_message = extract_chain.invoke({
        "request": "Teach prompt chaining to Python developers."
    })
    requirements = requirements_message.content
    print("REQUIREMENTS\n", requirements)

    # Second API call: the previous text becomes this step's input.
    draft_message = draft_chain.invoke({"requirements": requirements})
    first_draft = draft_message.content
    print("\nFIRST DRAFT\n", first_draft)

    # Third API call: polish the draft and print the final text.
    final_message = polish_chain.invoke({"draft": first_draft})
    final = final_message.content
    print("\nFINAL\n", final)
PY

def source(cell)
  value = cell["source"]
  value.is_a?(Array) ? value.join : value.to_s
end

def markdown(text)
  {"cell_type" => "markdown", "metadata" => {}, "source" => text}
end

def previous_heading(cells)
  cells.reverse_each do |cell|
    next unless cell["cell_type"] == "markdown"
    heading = source(cell).lines.find { |line| line.start_with?("#") }
    return heading&.sub(/^#+\s*/, "")&.strip if heading
  end
  "the example"
end

def requirements_for(code)
  requirements = []
  requirements << "`OPENAI_API_KEY` and network access" if code.include?("OPENAI_API_KEY") || code.include?("openai:gpt")
  requirements << "`LANGSMITH_API_KEY` and a LangSmith workspace" if code.include?("Client()") || code.include?("LANGSMITH")
  requirements << "a configured sandbox provider" if code.match?(/LangSmithSandbox|DaytonaSandbox|Modal/)
  requirements.empty? ? "No external credentials; this cell runs locally." : "Requires #{requirements.join("; ")}."
end

def syntax_notes(code)
  notes = []
  notes << "`import` loads code from another module; it does not call a model or run the workflow." if code.match?(/^import |^from /)
  notes << "`if` chooses one branch. Indented lines belong to that branch." if code.include?("if ")
  notes << "`def name(...):` defines a function. Its indented body runs only when the function is called." if code.include?("def ")
  notes << "`class Name(...):` defines a reusable type or a typed state/schema." if code.include?("class ")
  notes << "`@decorator` changes or registers the function immediately below it." if code.match?(/^\s*@/)
  notes << "A dictionary uses `{key: value}`; retrieve a value with `mapping[key]`." if code.include?("{") && code.include?("}")
  notes << "A list uses `[item1, item2]`; list order is preserved." if code.include?("[") && code.include?("]")
  notes << "An f-string such as `f\"{value}\"` inserts a Python value into text." if code.match?(/f["']/)
  notes << "A comprehension builds a collection by looping inside brackets." if code.match?(/\[[^\]]+ for .+ in /)
  notes << "`lambda` defines a small unnamed function used here as a callback." if code.include?("lambda ")
  notes << "`with` creates a context whose cleanup happens automatically when the indented block ends." if code.match?(/^\s*with /)
  notes << "Type hints after `:` describe expected values for readers and tools; Python does not enforce most hints by itself." if code.match?(/def .+\(.+:.+\)/) || code.include?("TypedDict")
  notes.first(6)
end

def framework_notes(code)
  notes = []
  notes << "`init_chat_model(...)` constructs a LangChain chat-model object; the API request happens later at `.invoke(...)`." if code.include?("init_chat_model")
  notes << "`ChatPromptTemplate.from_messages(...)` stores role-based message templates and their `{placeholders}`." if code.include?("ChatPromptTemplate")
  notes << "The `|` operator is overloaded by LangChain: `prompt | model` creates a runnable pipeline, not a mathematical operation." if code.match?(/\|\s*model/)
  notes << "`.invoke(input)` runs one synchronous request and returns a result; chat models return an `AIMessage`, whose text is available through `.content`." if code.include?(".invoke(")
  notes << "`create_agent(...)` builds the model/tool loop; the agent executes requested tools until it can return a final message." if code.include?("create_agent")
  notes << "`StateGraph(State)` creates a LangGraph builder around a shared state schema; nodes return partial state updates." if code.include?("StateGraph")
  notes << "`compile()` validates the graph and produces the runnable object used by `.invoke()` or `.stream()`." if code.include?(".compile(")
  notes << "`interrupt(...)` pauses a checkpointed graph; `Command(resume=...)` supplies the value when execution continues." if code.include?("interrupt(")
  notes << "`create_deep_agent(...)` adds planning, filesystem/context tools, summarization, and optional subagents over LangGraph." if code.include?("create_deep_agent")
  notes << "`Client.evaluate(...)` runs a target over a LangSmith dataset and records evaluator scores as an experiment." if code.include?(".evaluate(")
  notes.first(5)
end

def prompt_chaining_walkthrough
  <<~MD
    #{MARKER}

    **Purpose:** make three model calls in a fixed order: extract requirements,
    draft an answer, then polish it. Each step receives only the previous
    artifact it needs.

    **Python syntax used:**

    - `import os` makes Python's operating-system helpers available.
    - `from package import name` imports one named object from a package.
    - `os.getenv("OPENAI_API_KEY")` reads an environment variable. Missing keys
      return `None`, which behaves like `False` in the `if` condition.
    - Parentheses call a function. Keyword arguments such as `temperature=0`
      name the option being supplied.
    - `[...]` is a list. Each `("system", "...")` item is a two-value tuple.
    - `{request}`, `{requirements}`, and `{draft}` are prompt placeholders;
      they are replaced by values passed to `.invoke({...})`.
    - `{...}` passed to `invoke` is a dictionary mapping placeholder names to
      values.
    - The `|` symbol is overloaded by LangChain. Here it connects a prompt to a
      model and creates a runnable pipeline.
    - `.content` reads the text field from the returned `AIMessage`.

    **Execution order:**

    1. Import the three dependencies.
    2. Check for the API key. Without it, print a helpful message and stop.
    3. Construct one model object. `temperature=0` asks for conservative
       sampling; it is not a correctness guarantee.
    4. Build three independent prompt-to-model pipelines. Building them does
       not make an API request.
    5. Invoke `extract`; save its text as `requirements`.
    6. Insert those requirements into `draft`; save its text as `first_draft`.
    7. Insert the draft into `polish`; print the final text.

    **Cost:** when enabled, this cell makes three model API calls. Inspect the
    intermediate variables instead of treating the chain as a black box.
  MD
end

def walkthrough(code, heading)
  return prompt_chaining_walkthrough if code.include?("extract_chain.invoke") && code.include?("first_draft")

  syntax = syntax_notes(code)
  framework = framework_notes(code)
  bullets = (syntax + framework).map { |note| "- #{note}" }.join("\n")
  expected = if code.include?("print(")
    "The cell prints its main result or a clear skip message."
  elsif code.include?(".invoke(")
    "The final expression is the workflow or model result, which Jupyter displays."
  else
    "The cell defines or computes the #{heading.downcase}; inspect the final expression and named variables."
  end

  <<~MD
    #{MARKER}

    **Purpose:** run the **#{heading}** example. #{requirements_for(code)}

    **Syntax and API to notice:**

    #{bullets.empty? ? "- This cell uses ordinary Python assignments and function calls." : bullets}

    **How to read it:** Python executes top to bottom. Imports and definitions
    prepare names first; the calls near the bottom perform the actual work.
    #{expected}
  MD
end

def add_inline_comments(code)
  return code if code.include?("# `os` is part of Python's standard library")

  generated = [
    "# Import the dependencies used by this example.\n",
    "# Define the data shape and small operations before running them.\n",
    "# Configure the framework object; this line prepares it but may not execute it yet.\n",
    "# Execute the configured model or workflow with the input below.\n",
  ]
  lines = code.lines.reject { |line| generated.include?(line.lstrip) }
  output = []
  import_comment = false
  definition_comment = false
  construction_comment = false
  invocation_comment = false

  lines.each do |line|
    stripped = line.strip
    indent = line[/^\s*/]

    if !import_comment && stripped.match?(/^(from|import) /)
      output << "#{indent}# Import the dependencies used by this example.\n"
      import_comment = true
    elsif !definition_comment && stripped.match?(/^(class|def) /)
      output << "#{indent}# Define the data shape and small operations before running them.\n"
      definition_comment = true
    elsif !construction_comment && (stripped.match?(/^(agent|builder|graph|model)\s*=/) || stripped.start_with?("create_deep_agent("))
      output << "#{indent}# Configure the framework object; this line prepares it but may not execute it yet.\n"
      construction_comment = true
    elsif !invocation_comment && stripped.match?(/\.(invoke|stream|evaluate)\(/)
      output << "#{indent}# Execute the configured model or workflow with the input below.\n"
      invocation_comment = true
    end

    output << line
  end

  output.join
end


Dir[File.join(ROOT, "notebooks/**/*.ipynb")].sort.each do |path|
  notebook = JSON.parse(File.read(path))
  notebook["cells"].reject! do |cell|
    cell["cell_type"] == "markdown" && source(cell).start_with?(MARKER)
  end

  if path.end_with?("/01_core_patterns/01_prompt_chaining.ipynb")
    framework_index = notebook["cells"].index do |cell|
      cell["cell_type"] == "code" && source(cell).include?("requirements = extract.invoke")
    end
    notebook["cells"][framework_index]["source"] = PROMPT_CHAINING_CODE if framework_index
  end

  rebuilt = []
  notebook["cells"].each do |cell|
    if cell["cell_type"] == "code"
      code = source(cell).sub(/\A# CODE WALKTHROUGH\n.*?# END CODE WALKTHROUGH\n\n/m, "")
      cell["source"] = add_inline_comments(code)
    end
    rebuilt << cell
  end
  notebook["cells"] = rebuilt
  File.write(path, JSON.pretty_generate(notebook) + "\n")
end

puts "Kept comments short and placed them beside the relevant code."
