"""
agent.py — The agent brain: planning, tool selection, multi-step execution.

This uses LangChain's tool-calling agent pattern:
1. The LLM reads the user's prompt
2. It decides which tool (if any) to call, and with what input
3. The tool result goes back to the LLM
4. The LLM decides the next step — repeat until the task is done
5. The LLM returns a final answer

This loop IS the "agentic" part: planning + tool use + multi-step execution,
without any hardcoded if/else logic deciding what happens next.
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import ALL_TOOLS

load_dotenv()  # reads the .env file and loads API keys automatically

SYSTEM_PROMPT = """You are an execution agent. The user will give you a goal in plain English.

Your job:
1. Break the goal into the smallest number of concrete steps needed.
2. Use the tools available to you to actually complete each step — do not just describe what you would do.
3. If the user asks you to save or produce something, you MUST call write_file as your last step.
4. Once everything is done, give a short summary of what you did and where the output is.

Think step by step, but only show your final summary to the user, not your internal reasoning."""


def _build_llm():
    """
    Picks the LLM provider based on LLM_PROVIDER in .env.
    Supports 'anthropic' (Claude, paid) or 'gemini' (Google, has a free tier).
    Defaults to gemini since it's free to test with.
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set in .env"
            )
        return ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=0,
            anthropic_api_key=api_key,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set in .env"
            )
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key,
        )

    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER='{provider}'. Use 'anthropic' or 'gemini'."
        )


def build_agent(verbose: bool = True) -> AgentExecutor:
    llm = _build_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=verbose,       # prints each tool call + result live — great for demos
        max_iterations=8,      # safety cap so it can't loop forever
        return_intermediate_steps=True,
    )
    return executor


def run_task(goal: str, verbose: bool = True):
    """Run one goal through the agent and return (final_answer, steps_taken)."""
    executor = build_agent(verbose=verbose)
    result = executor.invoke({"input": goal})
    return result["output"], result.get("intermediate_steps", [])