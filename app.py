"""Streamlit demo for the Career Guidance Support Chatbot.

Run:  streamlit run app.py
The model is trained and saved by career_guidance_chatbot.ipynb (models/career_chatbot.joblib).
Colours and font are set in .streamlit/config.toml; STYLE below adds the chat layout.
"""
import json
import os
import time
import csv
from datetime import datetime

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

from chatbot_core import (INTENT_NAMES, INTENTS, MAX_QUERY_CHARS, METADATA_PATH, MODEL_PATH, CareerChatbot,
                          ConversationContext)

st.set_page_config(page_title="Career Guidance Chatbot", page_icon="🎓", layout="centered")

DATASET_URL = "https://huggingface.co/datasets/Pradeep016/career-guidance-qa-dataset"
AVATAR = ":material/school:"
STARTERS = [  # cards on the empty chat screen
    (":material/payments:", "How much do Software Engineers earn?"),
    (":material/psychology:", "What skills does a Data Analyst need?"),
    (":material/school:", "How do I start a career as a UX Researcher?"),
    (":material/explore:", "What is the best job for me?"),
]
EXAMPLES = [  # sidebar demo questions, one per feature
    (":material/short_text:", "ml engineer salary"),
    (":material/spellcheck:", "what skils does a data scienist need"),
    (":material/group:", "What is a Data Analyst, Data Scientist, Business Analyst?"),
    (":material/help:", "What is the average salary?"),
    (":material/list:", "list the jobs for computer science"),
    (":material/block:", "tell me a joke"),
]
META_LABELS = {"greeting": "greeting", "thanks": "thanks", "help": "help request",
               "recommend": "career recommendation", "list_careers": "career list"}
KIND_BADGES = {"clarify_role": ("Needs clarification", "orange"), "clarify_intent": ("Needs clarification", "orange"),
               "fallback": ("Fallback", "gray")}
THINKING_SECONDS = 1.0  # minimum time the indicator stays up; the model replies in well under that, so it would only flicker
THINKING_HTML = '<div class="thinking"><span></span><span></span><span></span><em>Thinking…</em></div>'
STYLE = """
<style>
/* Chat: the user's messages on the right in a tinted bubble, the assistant's on the left without a box */
[data-testid="stChatMessage"] {background: transparent; padding: .35rem 0;}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {flex-direction: row-reverse;}
[data-testid="stChatMessageAvatarUser"] {display: none;}
[aria-label="Chat message from user"] {flex: 0 1 auto; width: auto; max-width: 85%; margin: 0 0 0 auto; padding: .6rem 1rem;
    border-radius: 1.25rem 1.25rem .35rem 1.25rem; background: rgba(99, 102, 241, .13);}
[data-testid="stChatMessageAvatarCustom"] {background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff;}
[data-testid="stChatInput"] {box-shadow: 0 4px 18px rgba(15, 23, 42, .08);}

/* Fade-in animation for smooth streaming */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in-text {
    animation: fadeIn 0.4s ease-out forwards;
}

/* Thinking indicator */
.thinking {display: flex; align-items: center; gap: .4rem; opacity: .75; padding: .3rem 0;}
.thinking span {width: .5rem; height: .5rem; border-radius: 50%; background: currentColor;
                animation: thinking-bounce 1.2s infinite ease-in-out;}
.thinking span:nth-child(2) {animation-delay: .15s;}
.thinking span:nth-child(3) {animation-delay: .3s;}
.thinking em {margin-left: .3rem;}
@keyframes thinking-bounce {0%, 60%, 100% {transform: translateY(0); opacity: .35;}
                            30% {transform: translateY(-.35rem); opacity: 1;}}
@media (prefers-reduced-motion: reduce) {.thinking span {transform: none !important;}}

/* Empty chat: greeting and starter cards */
.st-key-welcome {padding-top: 7vh;}
.hero-icon {width: 4rem; height: 4rem; margin: 0 auto; border-radius: 1.25rem; display: grid; place-items: center;
            font-size: 2.1rem; background: linear-gradient(135deg, rgba(99, 102, 241, .2), rgba(139, 92, 246, .2));}
.st-key-starters button {min-height: 5.5rem; padding: .85rem 1rem; align-items: flex-start; justify-content: flex-start;}
.st-key-starters button > div {width: 100%; align-items: flex-start;}
.st-key-starters button > div > span {width: 100%; flex-direction: column; align-items: flex-start; gap: .5rem;}
.st-key-starters button p {text-align: left;}
.st-key-starters button:hover {border-color: #6366f1; background: rgba(99, 102, 241, .06);}

/* Follow-up chips and the sidebar question list */
[class*="st-key-chips"] button {border-radius: 1rem; min-height: 0; padding: .3rem .85rem; text-align: left;}
.st-key-examples button > div {justify-content: flex-start; text-align: left;}
</style>
"""


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
    """Queue a question (used by starter, example, suggestion and 'Did you mean' buttons)."""
    st.session_state.pending = question


def new_chat() -> None:
    """Archive current chat to history list or update existing, then reset for a fresh chat session."""
    if st.session_state.get("messages"):
        active_idx = st.session_state.get("active_session_index")
        first_q = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "Chat Session")
        title = (first_q[:25] + "...") if len(first_q) > 25 else first_q
        
        session_data = {
            "title": title,
            "messages": list(st.session_state.messages),
            "ctx": st.session_state.ctx
        }
        
        sessions = st.session_state.setdefault("sessions", [])
        
        if active_idx is not None and active_idx < len(sessions):
            # If we were viewing an old chat and modified it, update and move it to the top
            sessions.pop(active_idx)
            sessions.append(session_data)
        else:
            # Otherwise, it's a completely new chat session
            sessions.append(session_data)
            
    st.session_state.messages = []
    st.session_state.ctx = ConversationContext()
    st.session_state.active_session_index = None
    for key in list(st.session_state.keys()):
        if key.startswith("feedback_"):
            del st.session_state[key]

def load_past_session(idx: int) -> None:
    """Switch the active view to a selected past session from the history list."""
    session = st.session_state.sessions[idx]
    st.session_state.messages = session["messages"]
    st.session_state.ctx = session["ctx"]
    st.session_state.active_session_index = idx  # Track which session is currently open


def show_memory(slot) -> None:
    """The sidebar's remembered-career badge, or nothing when no career is remembered."""
    role = st.session_state.ctx.last_role
    if role:
        slot.badge(f"Remembering: {role}", icon=":material/bookmark:", color="violet",
                   help="Follow-up questions such as \"how much do they earn?\" refer to this career.")
    else:
        slot.empty()


def show_text(text: str) -> None:
    """Chat text as markdown, with "$" escaped so "$70,000 to $100,000" doesn't render as LaTeX."""
    st.markdown(text.replace("$", r"\$"))


def render_welcome(n_roles: int) -> None:
    """Empty-chat screen: a greeting and starter questions."""
    with st.container(key="welcome"):
        st.markdown('<div class="hero-icon">🎓</div>', unsafe_allow_html=True)
        st.header("What career are you curious about?", anchor=False, text_alignment="center")
        st.markdown(f"Ask about skills, salary, education, daily tasks, tools, challenges, work environment, career "
                    f"growth or how to get started, for **{n_roles} careers**. Typos, abbreviations and follow-ups "
                    "such as *\"how much do they earn?\"* are fine.", text_alignment="center")
        with st.container(key="starters"):
            cols = st.columns(2)
            for k, (icon, q) in enumerate(STARTERS):
                cols[k % 2].button(q, key=f"starter{k}", icon=icon, width="stretch", wrap=True, on_click=ask, args=(q,))


def render_details(d: dict, index: int, latest: bool) -> None:
    """Notes, confidence badge, model internals and follow-up buttons for one bot reply."""
    if d.get("corrections"):
        st.caption(":material/spellcheck: I read your question as: *" + d["normalized_query"] + "*")
    if d.get("used_memory") == "role":
        who = ", ".join(d.get("roles") or []) or d["role"]
        st.caption(f":material/history: Assuming you still mean **{who}** (from earlier in this conversation).")
    elif d.get("used_memory") == "intent":
        st.caption(f":material/history: Assuming you still want to know about **{INTENT_NAMES[d['intent']].lower()}**.")
    if d["kind"] != "invalid":
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            if d["kind"] == "answer":
                st.badge(f"{d['confidence']:.0%} confident", icon=":material/verified:",
                         color="green" if d["confidence"] >= 0.8 else "orange")
                st.caption(f"similarity {d['similarity']:.2f}", width="content")
            elif d["kind"] == "meta":
                st.badge(META_LABELS.get(d.get("meta_intent"), "conversation").capitalize(), icon=":material/forum:")
                st.caption(f"similarity {d['similarity']:.2f}", width="content")
            else:
                label, color = KIND_BADGES[d["kind"]]
                st.badge(label, icon=":material/info:", color=color,
                         help="The model was not confident enough to answer directly.")
            if d["kind"] != "meta":
                with st.popover("How the model decided", icon=":material/insights:", type="tertiary"):
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

    # --- Feature 2: Quick Feedback & Smooth Copy Popover ---
    feedback_key = f"feedback_{index}"
    if feedback_key not in st.session_state:
        st.session_state[feedback_key] = None

    def save_feedback(rating: str):
        st.session_state[feedback_key] = rating
        file_exists = os.path.isfile("feedback_log.csv")
        try:
            with open("feedback_log.csv", mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "User_Query", "Normalized_Query", "Rating"])
                user_q = st.session_state.messages[index - 1]["content"] if index > 0 and index < len(st.session_state.messages) else "N/A"
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_q, d.get("normalized_query", ""), rating])
        except Exception as e:
            print(f"Error saving feedback: {e}")

    col_fb1, col_fb2, col_copy, col_fb3 = st.columns([0.1, 0.1, 0.22, 0.58], vertical_alignment="center")
    with col_fb1:
        if st.session_state[feedback_key] is None:
            if st.button("👍", key=f"thumbs_up_{index}", help="Helpful"):
                save_feedback("helpful")
                st.rerun()
        else:
            st.markdown(":material/thumb_up:" if st.session_state[feedback_key] == "helpful" else ":material/thumb_down:")
    with col_fb2:
        if st.session_state[feedback_key] is None:
            if st.button("👎", key=f"thumbs_down_{index}", help="Not helpful"):
                save_feedback("not_helpful")
                st.rerun()
    with col_copy:
        with st.popover("Copy"):
            st.caption("Click icon to copy:")
            msg_content = st.session_state.messages[index]["content"] if index < len(st.session_state.messages) else ""
            st.code(msg_content, language="")
    with col_fb3:
        if st.session_state[feedback_key] is not None:
            v_text = "Logged as helpful." if st.session_state[feedback_key] == "helpful" else "Logged for improvement."
            st.caption(f":material/check_circle: {v_text}")

    actions = st.empty()
    if not latest:
        return
    with actions.container():
        if d.get("alternative"):
            st.button(f"Did you mean: {d['alternative'][1]}", key=f"alt{index}", icon=":material/lightbulb:",
                      on_click=ask, args=(d["alternative"][1],))
        if d.get("suggestions"):
            st.caption("You could also ask")
            with st.container(key=f"chips{index}", horizontal=True, gap="small"):
                for j, q in enumerate(d["suggestions"]):
                    st.button(q, key=f"sug{index}_{j}", icon=":material/arrow_forward:", wrap=True, on_click=ask,
                              args=(q,))


# ------------------------------------------------------------------ layout
st.html(STYLE)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH.name}`. Run `career_guidance_chatbot.ipynb` top to bottom "
             "first — it trains the model and saves it to the `models/` folder.")
    st.stop()
try:
    bot = load_bot()
except Exception as exc:
    st.error(f"Could not load the model: {exc}\n\nRe-run the notebook to regenerate it; the first run needs "
             "internet access to download the sentence-embedding model.")
    st.stop()

meta = load_metadata()
n_roles = bot.kb_["role"].nunique()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("ctx", ConversationContext())
st.session_state.setdefault("sessions", [])

with st.sidebar:
    st.subheader("🎓 Career Guide", anchor=False)
    st.caption(f"Career-guidance assistant for {n_roles} careers")
    st.button("New chat", icon=":material/edit_square:", width="stretch", on_click=new_chat, key="new_chat_btn")

    # --- Recent Conversations Sidebar List ---
    if st.session_state.sessions:
        st.caption("Recent Chats")
        for s_idx in reversed(range(len(st.session_state.sessions))):
            session = st.session_state.sessions[s_idx]
            if st.button(session["title"], key=f"history_session_{s_idx}", width="stretch", type="tertiary"):
                load_past_session(s_idx)
                st.rerun()
            st.markdown('<hr style="margin: 2px 0px 8px 0px; border-color: rgba(255, 255, 255, 0.08);">', unsafe_allow_html=True)
        st.divider()

    # --- Feature 1 & 4: Structured Career Report & Chat Export ---
    st.subheader("📄 Report Generator")
    
    report_content = "=== CAREER GUIDANCE ASSISTANT - SESSION REPORT ===\n\n"
    if st.session_state.get("messages"):
        for m in st.session_state.messages:
            role = "User" if m["role"] == "user" else "Assistant"
            report_content += f"[{role}]: {m['content']}\n\n"
    else:
        report_content += "No session history recorded yet."

    st.download_button(
        label="Download Career Report (.txt)",
        data=report_content,
        file_name="career_guidance_session_report.txt",
        mime="text/plain",
        icon=":material/description:",
        key="export_report_btn",
        disabled=not bool(st.session_state.get("messages"))
    )

    memory_note = st.empty()
    show_memory(memory_note)
    st.caption("Try asking")
    with st.container(key="examples", gap=None):
        for icon, q in EXAMPLES:
            st.button(q, icon=icon, type="tertiary", width="stretch", on_click=ask, args=(q,))
    st.divider()
    with st.expander("About the model", icon=":material/analytics:"):
        tm = meta.get("test_metrics", {})
        st.markdown(f"- Knowledge base: **{meta.get('kb_size', len(bot.kb_))}** Q&A pairs, "
                    f"**{meta.get('n_roles', n_roles)}** careers\n"
                    f"- Topic encoder: **{meta.get('encoder_label', bot.encoder)}**")
        if tm:
            st.markdown(f"- Held-out test (unseen wordings): role accuracy **{tm['role_accuracy']:.0%}**, "
                        f"topic accuracy **{tm['intent_accuracy']:.0%}**")
        real = meta.get("realistic_accuracy", {})
        if "ALL (64 questions)" in real:
            st.markdown(f"- Realistic questions handled correctly: **{real['ALL (64 questions)']:.0%}**")
    
    # --- Feature 3: Dynamic Local Philippine Salary & Tech Context (Covers all 54 Careers) ---
    with st.expander("🇵🇭 PH Salary & Tech Context", icon=":material/payments:"):
        st.markdown("Localized market estimates for the Philippines (PHP/month):")
        
        all_kb_roles = sorted(bot.kb_["role"].unique().tolist())
        selected_ph_role = st.selectbox("Select role for local rates:", all_kb_roles, key="ph_role_select")
        
        def get_ph_salary_estimate(role_name: str):
            lower_name = role_name.lower()
            if any(k in lower_name for k in ["ai", "machine learning", "cloud", "security", "sre", "data scientist"]):
                return {"entry": "₱40,000 - ₱65,000", "mid": "₱75,000 - ₱120,000", "senior": "₱130,000 - ₱200,000+"}
            elif any(k in lower_name for k in ["software", "developer", "engineer", "architect", "devops"]):
                return {"entry": "₱28,000 - ₱45,000", "mid": "₱55,000 - ₱95,000", "senior": "₱110,000 - ₱160,000+"}
            elif any(k in lower_name for k in ["analyst", "scientist", "specialist", "scrum", "product"]):
                return {"entry": "₱24,000 - ₱40,000", "mid": "₱48,000 - ₱85,000", "senior": "₱95,000 - ₱140,000+"}
            else:
                return {"entry": "₱20,000 - ₱32,000", "mid": "₱38,000 - ₱65,000", "senior": "₱75,000 - ₱110,000+"}

        r_data = get_ph_salary_estimate(selected_ph_role)
        st.markdown(f"""
        * **Entry-Level:** {r_data['entry']}
        * **Mid-Level:** {r_data['mid']}
        * **Senior-Level:** {r_data['senior']}
        """)
        st.caption("Values reflect standard Philippine IT industry averages (BGC, Ortigas, Cebu, and Remote).")

    with st.expander(f"Careers covered ({n_roles})", icon=":material/work:"):
        st.write(", ".join(sorted(bot.kb_["role"].unique())))
    with st.expander("Topics", icon=":material/topic:"):
        st.write(", ".join(INTENT_NAMES[i] for i in INTENTS))
    st.caption(f"Answers come from the [career-guidance QA dataset]({DATASET_URL}) via the NLP model trained in "
               "`career_guidance_chatbot.ipynb`.")

query = st.chat_input("Ask about any career…", max_chars=MAX_QUERY_CHARS) or st.session_state.pop("pending", None)

if not st.session_state.messages and not query:
    render_welcome(n_roles)

last = len(st.session_state.messages) - 1

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"], avatar=AVATAR if m["role"] == "assistant" else None):
        show_text(m["content"])
        if m["role"] == "assistant":
            render_details(m["details"], i, latest=(i == last and not query))

if query:
    # DO NOT reset active_session_index here so we can update the existing session properly!

    with st.chat_message("user"):
        show_text(query)
    with st.chat_message("assistant", avatar=AVATAR):
        body = st.empty()
        body.markdown(THINKING_HTML, unsafe_allow_html=True)
        started = time.monotonic()
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
        
        time.sleep(max(0.0, THINKING_SECONDS - (time.monotonic() - started)))
        
        # --- Smooth, Fading Word-by-Word Animation ---
        streamed_text = ""
        words = text.split(" ")
        for w_idx, word in enumerate(words):
            streamed_text += word + (" " if w_idx < len(words) - 1 else "")
            body.markdown(f'<div class="fade-in-text">{streamed_text.replace("$", r"\$")}</div>', unsafe_allow_html=True)
            time.sleep(0.04)

        st.session_state.messages += [{"role": "user", "content": query},
                                      {"role": "assistant", "content": text, "details": details}]
        with body:
            show_text(text)
        render_details(details, len(st.session_state.messages) - 1, latest=True)
    show_memory(memory_note)
    st.rerun()