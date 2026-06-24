"""
main.py — CLI entrypoint.

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python main.py "Research the top 3 AI agent frameworks and save a comparison to frameworks.md"

Or run with no argument for interactive mode.
"""

import sys
from agent import run_task
from recommender import log_goal, recommend_next_actions


def main():
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])
    else:
        print("Prompt-to-Action Agent — type a goal and watch it get executed.")
        print("Example: Research the top 3 AI agent frameworks and save a comparison to frameworks.md\n")
        goal = input("What do you want done? > ").strip()

    if not goal:
        print("No goal given, exiting.")
        return

    print(f"\n>>> Goal: {goal}\n")
    print(">>> Agent is planning and executing (tool calls will print below)...\n")

    final_answer, steps = run_task(goal, verbose=True)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(final_answer)
    print(f"\n({len(steps)} tool call(s) made)")

    # Log this goal and surface relevant next-action suggestions
    log_goal(goal)
    suggestions = recommend_next_actions(goal)
    if suggestions:
        print("\n" + "-" * 60)
        print("You might also want to:")
        for s in suggestions:
            print(f"  • {s}")


if __name__ == "__main__":
    main()
