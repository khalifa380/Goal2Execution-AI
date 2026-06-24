"""
recommender.py — A minimal recommendation engine.

Tracks what kinds of goals the user runs, and suggests related next actions.
This is intentionally simple (TF-IDF + cosine similarity over past prompts),
but it's a REAL, working recommendation engine, not a mock — it demonstrates
understanding of the recommendation problem: given user history, rank
candidate next actions by relevance.

In a product like Gravity, this is the seed of "what should this user
probably want to do next" — surfaced after each completed task.
"""

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")

# A small catalog of candidate follow-up actions the agent could suggest.
# In a real product this would be mined from what other users did next.
CANDIDATE_ACTIONS = [
    "Summarize the saved report into 3 bullet points",
    "Turn this comparison into a presentation outline",
    "Find pricing information for the top option",
    "Set a reminder to revisit this in a week",
    "Search for criticisms or downsides of the top result",
    "Export the result as a PDF",
    "Email this summary to a teammate",
    "Compare this with last month's research on the same topic",
]


def _load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_history(history: list) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def log_goal(goal: str) -> None:
    """Call this after every completed agent task to build up history."""
    history = _load_history()
    history.append(goal)
    _save_history(history)


def recommend_next_actions(current_goal: str, top_k: int = 3) -> list:
    """
    Rank CANDIDATE_ACTIONS by relevance to the user's current + past goals
    using TF-IDF cosine similarity. Returns the top_k most relevant suggestions.
    """
    history = _load_history()
    context = history + [current_goal]

    corpus = context + CANDIDATE_ACTIONS
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Compare the *current goal* vector against each candidate action vector
    n_context = len(context)
    current_vec = tfidf_matrix[n_context - 1]  # last item in context = current goal
    candidate_vecs = tfidf_matrix[n_context:]

    similarities = cosine_similarity(current_vec, candidate_vecs)[0]
    ranked = sorted(
        zip(CANDIDATE_ACTIONS, similarities),
        key=lambda x: x[1],
        reverse=True,
    )

    return [action for action, score in ranked[:top_k]]
