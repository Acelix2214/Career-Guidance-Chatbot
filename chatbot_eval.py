"""Evaluation helpers used by the notebook: frozen test sets, realistic-query scoring,
multi-turn dialogue scoring and latency measurement."""
from __future__ import annotations

import hashlib
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import chatbot_core as cc

DATA_DIR = cc.PROJECT_DIR / "data"
QUERIES_PATH = DATA_DIR / "realistic_queries.csv"
CONVERSATIONS_PATH = DATA_DIR / "conversations.csv"


def sha256(path: Path) -> str:
    """Hash of a file, printed in the notebook so any later edit of a frozen test set is visible."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_eval_sets() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """The hand-written realistic queries and scripted dialogues (written before any tuning)."""
    q = pd.read_csv(QUERIES_PATH, dtype=str, keep_default_na=False)
    c = pd.read_csv(CONVERSATIONS_PATH, dtype=str, keep_default_na=False)
    c["turn"] = c["turn"].astype(int)
    return q, c, {p.name: sha256(p) for p in (QUERIES_PATH, CONVERSATIONS_PATH)}


EXTRA_TEMPLATES_PATH = DATA_DIR / "extra_question_templates.csv"


def load_extra_templates() -> pd.DataFrame:
    """Hand-written extra question wordings (``{role}`` / ``{roles}`` placeholders), one row per wording."""
    return pd.read_csv(EXTRA_TEMPLATES_PATH, dtype=str, keep_default_na=False)


def expand_templates(templates: pd.DataFrame, roles) -> pd.DataFrame:
    """Fill every wording with every career name -> extra (role, intent, question) training rows."""
    rows = []
    for role in roles:
        base = cc.role_base_name(role)
        for t in templates.itertuples(index=False):
            rows.append({"role": role, "intent": t.intent, "template": t.template,
                         "question": t.template.replace("{roles}", base + "s").replace("{role}", base)})
    return pd.DataFrame(rows)


def content_words(text: str, roles=()) -> set[str]:
    """Lower-case content words with career names masked as 'role' (for wording-overlap checks)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    t = text.lower().replace("'s", "")
    t = t.replace("{roles}", " role ").replace("{role}", " role ").replace("<role>", " role ")
    for r in sorted(roles, key=len, reverse=True):
        t = t.replace(r.lower(), " role ")
    return {w for w in cc._WORD_RE.findall(t)} - set(ENGLISH_STOP_WORDS)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def is_correct(resp: cc.ChatResponse, expected_kind: str, role: str, intent: str) -> bool:
    """answer -> right career AND topic; clarify_role -> asks for the career; no_answer -> does not answer."""
    if expected_kind == "answer":
        return resp.kind == "answer" and resp.role == role and resp.intent == intent
    if expected_kind == "no_answer":
        return resp.kind != "answer"
    return resp.kind == expected_kind


def evaluate_queries(bot: cc.CareerChatbot, queries: pd.DataFrame, use_normalizer: bool | None = None) -> pd.DataFrame:
    """Run every query through ``bot.respond`` (the app's code path) and score it."""
    saved = bot.use_normalizer
    if use_normalizer is not None:
        bot.use_normalizer = use_normalizer
    try:
        rows = []
        for r in queries.itertuples(index=False):
            resp = bot.respond(r.query)
            rows.append({"id": r.id, "category": r.category, "query": r.query, "expected": r.expected_kind,
                         "decision": resp.kind, "pred_role": resp.role, "pred_intent": resp.intent,
                         "exp_role": r.expected_role, "exp_intent": r.expected_intent,
                         "normalized": resp.normalized_query, "did_you_mean": resp.alternative is not None,
                         "correct": is_correct(resp, r.expected_kind, r.expected_role, r.expected_intent)})
        return pd.DataFrame(rows)
    finally:
        bot.use_normalizer = saved


def run_dialogues(bot: cc.CareerChatbot, conversations: pd.DataFrame, memory: bool = True) -> pd.DataFrame:
    """Replay each scripted dialogue turn by turn, passing the context only when ``memory`` is on."""
    rows = []
    for dialog_id, turns in conversations.sort_values(["dialog_id", "turn"]).groupby("dialog_id", sort=True):
        ctx = cc.ConversationContext() if memory else None
        for r in turns.itertuples(index=False):
            resp = bot.respond(r.query, ctx)
            if memory:
                ctx = resp.context
            rows.append({"dialog_id": dialog_id, "turn": r.turn, "query": r.query, "expected": r.expected_kind,
                         "decision": resp.kind, "pred_role": resp.role, "pred_intent": resp.intent,
                         "exp_role": r.expected_role, "exp_intent": r.expected_intent, "used_memory": resp.used_memory,
                         "correct": is_correct(resp, r.expected_kind, r.expected_role, r.expected_intent)})
    return pd.DataFrame(rows)


def reply_latency(bot: cc.CareerChatbot, queries, repeats: int = 3) -> dict[str, float]:
    """Warm per-message latency of ``respond`` in milliseconds (median and 95th percentile)."""
    bot.respond("warm up")
    times = []
    for _ in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            bot.respond(q)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"median_ms": statistics.median(times), "p95_ms": times[int(0.95 * (len(times) - 1))]}


_COLD_START = (
    "import time; t0 = time.perf_counter(); import chatbot_core as cc; t1 = time.perf_counter(); "
    "b = cc.CareerChatbot.load(r'{path}'); t2 = time.perf_counter(); b.warmup(); t3 = time.perf_counter(); "
    "print(f'{{t1-t0:.2f}} {{t2-t1:.2f}} {{t3-t2:.2f}}')"
)


def cold_start(model_path: Path) -> dict[str, float]:
    """Seconds to import, load and warm up a model in a fresh Python process (what the app pays at startup)."""
    out = subprocess.run([sys.executable, "-c", _COLD_START.format(path=model_path)], cwd=cc.PROJECT_DIR,
                         capture_output=True, text=True, timeout=600, check=True).stdout.split()
    imp, load, warm = map(float, out[-3:])
    return {"import_s": imp, "load_s": load, "warmup_s": warm, "total_s": imp + load + warm}
