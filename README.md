# Prompt-to-Action Agent

A small, working agentic AI system: type one plain-English goal, and an LLM agent
**plans the steps, calls real tools, executes multi-step work, and produces a finished
result** — no manual configuration per task.

Built to explore the same core problem as products like Gravity: collapsing "prompt →
working result" into one step, instead of a manual multi-tool workflow.

## What it actually does

```
$ python main.py "Research the top 3 AI agent frameworks and save a comparison to frameworks.md"

>>> Agent is planning and executing...
  [tool call] web_search("top AI agent frameworks 2026")
  [tool call] write_file("frameworks.md ||| # Top AI Agent Frameworks\n\n1. LangChain — ...")

RESULT
I researched the top 3 AI agent frameworks and saved a comparison to
outputs/frameworks.md, covering LangChain, AutoGen, and CrewAI.

You might also want to:
  • Turn this comparison into a presentation outline
  • Export the result as a PDF
  • Search for criticisms or downsides of the top result
```

One prompt in. Real file out. No step-by-step instructions from the user.

## How it works

| Piece | What it does | File |
|---|---|---|
| **Agent loop** | LLM plans steps, decides which tool to call, observes the result, decides the next step, repeats until done | `agent.py` |
| **Tools** | Web search, calculator, sandboxed file writer — the actions the agent can actually take | `tools.py` |
| **Recommender** | TF-IDF similarity over goal history to suggest relevant next actions after each task | `recommender.py` |
| **CI** | Runs lint + unit tests on every push, so a broken tool never reaches the agent | `.github/workflows/ci.yml` |
| **Tests** | Real unit tests for tools and the recommender (not just smoke tests) | `tests/` |

This is built with **LangChain's tool-calling agent pattern**: there is no hardcoded
`if/else` deciding what happens next. The LLM itself decides which tool to use, in
what order, based on the goal — that decision loop is what makes it "agentic" rather
than a fixed pipeline.

## Setup

```bash
git clone <this-repo>
cd prompt-to-action-agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python main.py "your goal here"
```

Or run with no arguments for interactive mode.

## Running tests

```bash
pip install pytest flake8
pytest tests/ -v
```

## Design notes / what I'd improve next

- **Tools are intentionally simple** (search, calculator, file write) to keep the
  agent loop the focus, not the tool catalog. Adding tools (email, calendar, API
  calls to other services) is a matter of writing one more `@tool`-decorated function.
- **The recommender is a real but simple model** (TF-IDF + cosine similarity over
  goal history). The next step would be replacing the static candidate list with
  ones mined from actual usage logs.
- **Safety**: the file-writer tool is sandboxed to one `outputs/` folder and strips
  directory traversal attempts via `os.path.basename`. The calculator uses a
  restricted `eval` with no builtins. Neither is production-hardened, but both
  reflect thinking about what a tool-using agent can go wrong and do.
- **No persistent memory across sessions yet** beyond the goal history file —
  a real product would need a proper store (DB, not JSON) and per-user isolation.

## Why I built this

I'm certified in the Claude API, MCP, Claude Code, and Agent Skills, but certificates
aren't proof of building. This is — a working, tested, CI-checked agent that takes
one prompt and actually does the work, end to end.
