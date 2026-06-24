"""
app.py — A simple web interface for the Prompt-to-Action Agent.

Run this instead of main.py if you want a browser UI instead of the terminal.

    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import io
import os
import contextlib
from flask import Flask, request, jsonify, render_template, send_from_directory

from agent import run_task
from recommender import log_goal, recommend_next_actions
from tools import OUTPUT_DIR

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_agent():
    data = request.get_json(silent=True) or {}
    goal = (data.get("goal") or "").strip()

    if not goal:
        return jsonify({"error": "Please type a goal first."}), 400

    # Capture everything the agent prints (its tool calls / reasoning trace)
    # so we can show it in the browser, not just the terminal.
    trace_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(trace_buffer):
            final_answer, steps = run_task(goal, verbose=True)
    except Exception as e:
        return jsonify({"error": f"The agent hit an error: {e}"}), 500

    log_goal(goal)
    suggestions = recommend_next_actions(goal)

    # Figure out if a file was written during this run, so we can offer it for download
    saved_files = []
    for _, observation in steps:
        if isinstance(observation, str) and "Saved successfully to" in observation:
            path = observation.split("Saved successfully to", 1)[1].strip()
            saved_files.append(os.path.basename(path))

    return jsonify({
        "trace": trace_buffer.getvalue(),
        "answer": final_answer,
        "tool_calls": len(steps),
        "suggestions": suggestions,
        "saved_files": saved_files,
    })


@app.route("/outputs/<path:filename>")
def download_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    print("\nOpen this in your browser: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
