"""
tools.py — Real tools the agent can call.

Each tool is a plain Python function wrapped with LangChain's @tool decorator.
The agent decides WHICH tool to call and WHEN, based on the user's prompt.
This is the "tool use" piece the JD asks about.
"""

import os
import math
from langchain.tools import tool

# Where the agent is allowed to write files (sandboxed to one folder)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _search_tavily(query: str) -> str:
    """Primary search: Tavily, a search API built for AI agents (free tier)."""
    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")

    client = TavilyClient(api_key=api_key)
    response = client.search(query, max_results=5)
    results = response.get("results", [])
    if not results:
        return "No results found."

    formatted = []
    for r in results:
        formatted.append(f"- {r.get('title', '')}: {r.get('content', '')}")
    return "\n".join(formatted)


def _search_ddgs(query: str) -> str:
    """Fallback search: DuckDuckGo via the ddgs package (no key needed, less reliable)."""
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return "No results found."

    formatted = []
    for r in results:
        formatted.append(f"- {r.get('title', '')}: {r.get('body', '')}")
    return "\n".join(formatted)


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information on a topic.
    Use this when you need facts, comparisons, or anything that
    might have changed recently. Input should be a short search query.
    """
    # Prefer Tavily (purpose-built for agents, free tier) if a key is set;
    # otherwise fall back to DuckDuckGo so the tool still works with no setup.
    if os.environ.get("TAVILY_API_KEY"):
        try:
            return _search_tavily(query)
        except Exception as e:
            return f"Tavily search failed ({e}); no fallback succeeded."

    try:
        return _search_ddgs(query)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic math expression, e.g. '12 * 7 + 3'.
    Use this for any arithmetic instead of guessing the answer yourself.
    """
    try:
        # Restricted eval: only math operators, no builtins/imports allowed
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculation failed: {e}"


@tool
def write_file(filename_and_content: str) -> str:
    """
    Save content to a file. Input MUST be in the exact format:
    'filename.txt ||| the full content to write'
    Use this as the final step when the user asks you to save,
    write, or produce a report/document.
    """
    try:
        if "|||" not in filename_and_content:
            return "Error: input must be 'filename.ext ||| content'"
        filename, content = filename_and_content.split("|||", 1)
        filename = filename.strip()
        content = content.strip()

        # Prevent path traversal — keep all writes inside OUTPUT_DIR
        safe_name = os.path.basename(filename)
        path = os.path.join(OUTPUT_DIR, safe_name)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Saved successfully to {path}"
    except Exception as e:
        return f"Write failed: {e}"


ALL_TOOLS = [web_search, calculator, write_file]