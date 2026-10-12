"""Streamlit demo for the Career Guidance Support Chatbot.

Run:  streamlit run app.py
The model is trained and saved by career_guidance_chatbot.ipynb (models/career_chatbot.joblib).
"""
import json
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

from chatbot_core import INTENT_NAMES, INTENTS, MAX_QUERY_CHARS, METADATA_PATH, MODEL_PATH, CareerChatbot, get_encoder

st.set_page_config(page_title="Career Guidance Chatbot", page_icon="🎓", layout="centered")

EXAMPLES = [
    "What skills does a Data Analyst need?",
    "How much do Software Engineers earn?",
    "How do I start a career as a UX Researcher?",
    "What does a typical day look like for a Product Owner?",
]
KIND_LABELS = {"answer": "Answer", "clarify_role": "Needs clarification",
               "clarify_intent": "Needs clarification", "fallback": "Fallback", "invalid": "Invalid input"}


@st.cache_resource(show_spinner="Loading the NLP model…")
def load_bot():
    bot = CareerChatbot.load(MODEL_PATH)
    if bot.uses_embeddings:
        get_encoder()  # load the sentence encoder once, not on the first question
    return bot


def load_metadata():
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def render_details(d):
    """Confidence indicator + model internals for one bot reply."""
    if d["kind"] == "answer":
        st.progress(min(max(d["confidence"], 0.0), 1.0),
                    text=f"Confidence {d['confidence']:.0%} · similarity {d['similarity']:.2f}")
    elif d["kind"] != "invalid":
        st.caption(f"{KIND_LABELS[d['kind']]} — the model was not confident enough to answer directly.")
    if d["kind"] != "invalid":
        with st.expander("How the model decided"):
            st.markdown(f"**Predicted role:** {d['role']} ({d['role_conf']:.0%})  \n"
                        f"**Predicted topic:** {INTENT_NAMES.get(d['intent'], d['intent'])} ({d['intent_conf']:.0%})  \n"
                        f"**Best knowledge-base similarity:** {d['similarity']:.2f}")
            if d["matched_question"]:
                st.markdown(f"**Matched stored question:** _{d['matched_question']}_")
            st.markdown("**Top roles:** " + ", ".join(f"{r} ({p:.0%})" for r, p in d["role_candidates"]))
            st.markdown("**Top topics:** " + ", ".join(f"{INTENT_NAMES[i]} ({p:.0%})" for i, p in d["intent_candidates"]))


# ------------------------------------------------------------------ layout
st.title("🎓 Career Guidance Chatbot")
st.write("Ask about a career's skills, salary, education, daily tasks, tools, challenges, work environment, "
         "career growth or how to get started. Answers come from the "
         "[career-guidance QA dataset](https://huggingface.co/datasets/Pradeep016/career-guidance-qa-dataset) "
         "via the NLP model trained in `career_guidance_chatbot.ipynb`.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH.name}`. Run `career_guidance_chatbot.ipynb` top to bottom "
             "first — it trains the model and saves it to the `models/` folder.")
    st.stop()
try:
    bot = load_bot()
except Exception as exc:  # corrupted file, version mismatch, or encoder download failure
    st.error(f"Could not load the model: {exc}\n\nRe-run the notebook to regenerate it; the first run needs "
             "internet access to download the sentence-embedding model.")
    st.stop()

meta = load_metadata()
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("Try an example")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    if st.button("Clear conversation", type="secondary", use_container_width=True):
        st.session_state.messages = []
    st.divider()
    st.subheader("About the model")
    tm = meta.get("test_metrics", {})
    st.markdown(f"- Knowledge base: **{meta.get('kb_size', len(bot.kb_))}** Q&A pairs, "
                f"**{meta.get('n_roles', bot.kb_['role'].nunique())}** careers\n"
                f"- Intent features: **{meta.get('intent_features', bot.intent_features)}**")
    if tm:
        st.markdown(f"- Held-out test (unseen wordings): role accuracy **{tm['role_accuracy']:.0%}**, "
                    f"intent accuracy **{tm['intent_accuracy']:.0%}**")
    with st.expander(f"Careers covered ({bot.kb_['role'].nunique()})"):
        st.write(", ".join(sorted(bot.kb_["role"].unique())))
    with st.expander("Topics"):
        st.write(", ".join(INTENT_NAMES[i] for i in INTENTS))

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_details(m["details"])

query = st.chat_input("Ask a career question…", max_chars=MAX_QUERY_CHARS) or st.session_state.pop("pending", None)
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        try:
            r = bot.respond(query)
            details = {"kind": r.kind, "confidence": r.confidence, "similarity": r.similarity, "role": r.role,
                       "role_conf": r.role_conf, "intent": r.intent, "intent_conf": r.intent_conf,
                       "matched_question": r.matched_question, "role_candidates": r.role_candidates,
                       "intent_candidates": r.intent_candidates}
            text = r.text
        except Exception as exc:
            text, details = f"Sorry, something went wrong while processing your question ({exc}).", {"kind": "invalid"}
        st.markdown(text)
        render_details(details)
    st.session_state.messages.append({"role": "assistant", "content": text, "details": details})
