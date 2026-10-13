"""Streamlit demo for the Career Guidance Support Chatbot.

Run:  streamlit run app.py
The model is trained and saved by career_guidance_chatbot.ipynb (models/career_chatbot.joblib).
"""
import json
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

from chatbot_core import (INTENT_NAMES, INTENTS, MAX_QUERY_CHARS, METADATA_PATH, MODEL_PATH, CareerChatbot,
                          ConversationContext)

st.set_page_config(page_title="Career Guidance Chatbot", page_icon="🎓", layout="centered")

EXAMPLES = [
    "What skills does a Data Analyst need?",
    "How much do Software Engineers earn?",
    "How do I start a career as a UX Researcher?",
    "ml engineer salary",
]
META_LABELS = {"greeting": "greeting", "thanks": "thanks", "help": "help request",
               "recommend": "career recommendation", "list_careers": "career list"}
KIND_LABELS = {"answer": "Answer", "meta": "Conversation", "clarify_role": "Needs clarification",
               "clarify_intent": "Needs clarification", "fallback": "Fallback", "invalid": "Invalid input"}


@st.cache_resource(show_spinner="Loading the NLP model…")
def load_bot() -> CareerChatbot:
    """Load the saved model once per server; ``warmup`` loads its own encoder and checks the fingerprint."""
    return CareerChatbot.load(MODEL_PATH).warmup()


def load_metadata() -> dict:
    """Model settings and evaluation results written by the notebook (empty if unavailable)."""
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def ask(question: str) -> None:
    """Queue a question (used by example, suggestion and 'Did you mean' buttons)."""
    st.session_state.pending = question


def clear() -> None:
    st.session_state.messages = []
    st.session_state.ctx = ConversationContext()


def render_details(d: dict, index: int, latest: bool) -> None:
    """Notes, confidence indicator, model internals and follow-up buttons for one bot reply."""
    if d.get("corrections"):
        st.caption("I read your question as: *" + d["normalized_query"] + "*")
    if d.get("used_memory") == "role":
        who = ", ".join(d.get("roles") or []) or d["role"]
        st.caption(f"Assuming you still mean **{who}** (from earlier in this conversation).")
    elif d.get("used_memory") == "intent":
        st.caption(f"Assuming you still want to know about **{INTENT_NAMES[d['intent']].lower()}**.")
    if d["kind"] == "answer":
        st.progress(min(max(d["confidence"], 0.0), 1.0),
                    text=f"Confidence {d['confidence']:.0%} · similarity {d['similarity']:.2f}")
    elif d["kind"] == "meta":
        st.caption(f"Recognised as a {META_LABELS.get(d.get('meta_intent'), 'conversation')} "
                   f"(similarity {d['similarity']:.2f}).")
    elif d["kind"] != "invalid":
        st.caption(f"{KIND_LABELS[d['kind']]} — the model was not confident enough to answer directly.")
    if d["kind"] not in ("invalid", "meta"):
        with st.expander("How the model decided"):
            multi = len(d.get("roles") or []) > 1
            if multi:
                st.markdown(f"**Careers answered ({len(d['roles'])}):** " + ", ".join(d["roles"]) + "  \n"
                            "*Each listed career was recognised separately by the role classifier.*")
            st.markdown((f"**Predicted role:** {d['role']} ({d['role_conf']:.0%})  \n" if not multi else "") +
                        f"**Predicted topic:** {INTENT_NAMES.get(d['intent'], d['intent'])} ({d['intent_conf']:.0%})  \n"
                        f"**Best knowledge-base similarity:** {d['similarity']:.2f}")
            if d.get("matched_question"):
                st.markdown(f"**Matched stored question:** _{d['matched_question']}_")
            if not multi:
                st.markdown("**Top roles:** " + ", ".join(f"{r} ({p:.0%})" for r, p in d["role_candidates"]))
            st.markdown("**Top topics:** " + ", ".join(f"{INTENT_NAMES[i]} ({p:.0%})" for i, p in d["intent_candidates"]))
    if not latest:
        return
    if d.get("alternative"):
        st.button(f"Did you mean: {d['alternative'][1]}", key=f"alt{index}", on_click=ask, args=(d["alternative"][1],))
    if d.get("suggestions"):
        st.caption("You could also ask:")
        for j, q in enumerate(d["suggestions"]):
            st.button(q, key=f"sug{index}_{j}", on_click=ask, args=(q,))


# ------------------------------------------------------------------ layout
st.title("🎓 Career Guidance Chatbot")
st.write("Ask about a career's skills, salary, education, daily tasks, tools, challenges, work environment, "
         "career growth or how to get started. Follow-up questions such as *\"how much do they earn?\"* refer to "
         "the career you asked about last. Answers come from the "
         "[career-guidance QA dataset](https://huggingface.co/datasets/Pradeep016/career-guidance-qa-dataset) "
         "via the NLP model trained in `career_guidance_chatbot.ipynb`.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH.name}`. Run `career_guidance_chatbot.ipynb` top to bottom "
             "first — it trains the model and saves it to the `models/` folder.")
    st.stop()
try:
    bot = load_bot()
except Exception as exc:  # corrupted file, version or encoder mismatch, or encoder download failure
    st.error(f"Could not load the model: {exc}\n\nRe-run the notebook to regenerate it; the first run needs "
             "internet access to download the sentence-embedding model.")
    st.stop()

meta = load_metadata()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("ctx", ConversationContext())

with st.sidebar:
    st.header("Try an example")
    for q in EXAMPLES:
        st.button(q, use_container_width=True, on_click=ask, args=(q,))
    st.button("Clear conversation", type="secondary", use_container_width=True, on_click=clear)
    st.divider()
    st.subheader("About the model")
    tm = meta.get("test_metrics", {})
    st.markdown(f"- Knowledge base: **{meta.get('kb_size', len(bot.kb_))}** Q&A pairs, "
                f"**{meta.get('n_roles', bot.kb_['role'].nunique())}** careers\n"
                f"- Topic encoder: **{meta.get('encoder_label', bot.encoder)}**")
    if tm:
        st.markdown(f"- Held-out test (unseen wordings): role accuracy **{tm['role_accuracy']:.0%}**, "
                    f"topic accuracy **{tm['intent_accuracy']:.0%}**")
    real = meta.get("realistic_accuracy", {})
    if "ALL (64 questions)" in real:
        st.markdown(f"- Realistic questions handled correctly: **{real['ALL (64 questions)']:.0%}**")
    ctx = st.session_state.ctx
    if ctx.last_role:
        st.caption(f"Remembered career: {ctx.last_role}")
    with st.expander(f"Careers covered ({bot.kb_['role'].nunique()})"):
        st.write(", ".join(sorted(bot.kb_["role"].unique())))
    with st.expander("Topics"):
        st.write(", ".join(INTENT_NAMES[i] for i in INTENTS))

query = st.chat_input("Ask a career question…", max_chars=MAX_QUERY_CHARS) or st.session_state.pop("pending", None)
if query:
    try:
        r = bot.respond(query, st.session_state.ctx)
        st.session_state.ctx = r.context or st.session_state.ctx
        details = {"kind": r.kind, "confidence": r.confidence, "similarity": r.similarity, "role": r.role,
                   "role_conf": r.role_conf, "intent": r.intent, "intent_conf": r.intent_conf,
                   "matched_question": r.matched_question, "role_candidates": r.role_candidates,
                   "intent_candidates": r.intent_candidates, "corrections": r.corrections,
                   "normalized_query": r.normalized_query, "used_memory": r.used_memory,
                   "alternative": r.alternative, "roles": r.roles, "meta_intent": r.meta_intent,
                   "suggestions": (r.suggestions if r.kind == "meta" else
                                   bot.suggest_followups(r.role, exclude=[r.intent])
                                   if r.kind == "answer" and len(r.roles) < 2 else [])}
        text = r.text
    except Exception as exc:
        text, details = f"Sorry, something went wrong while processing your question ({exc}).", {"kind": "invalid"}
    st.session_state.messages += [{"role": "user", "content": query},
                                  {"role": "assistant", "content": text, "details": details}]

last = len(st.session_state.messages) - 1
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_details(m["details"], i, latest=(i == last))
