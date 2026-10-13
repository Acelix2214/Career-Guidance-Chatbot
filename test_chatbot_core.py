"""Self-checks for chatbot_core and the saved model. Run: python test_chatbot_core.py"""
from pathlib import Path

import pandas as pd

import chatbot_core as cc

V1_MODEL = Path(__file__).resolve().parent / "backup" / "v1.0" / "models" / "career_chatbot.joblib"


def test_preprocessing():
    p = cc.TextPreprocessor(remove_stopwords=True, stem=False)
    assert p("Where do Data Scientists WORK?") == "where do data scientists work"  # intent words kept
    assert p("What's the salary?") == "what salary"
    assert cc.TextPreprocessor(True, True)("earns") == "earn"
    assert p("!!! ???") == ""


def test_intent_labels():
    df = pd.DataFrame({"role": ["A"] * 30 + ["B"] * 30, "question": [f"q{i}" for i in range(60)], "answer": "x"})
    out = cc.add_intent_labels(df)
    assert out.intent.iloc[0] == "role_overview" and out.intent.iloc[29] == "getting_started"
    assert out.intent.iloc[30] == "role_overview" and list(out.paraphrase_slot[:4]) == [0, 1, 2, 0]


def test_query_normalizer():
    kb = pd.DataFrame({"role": ["Site Reliability Engineer (SRE)", "Machine Learning Engineer"],
                       "question": ["What skills does a Site Reliability Engineer need?",
                                    "How much do Machine Learning Engineers earn?"],
                       "answer": ["Linux and monitoring.", "Whether they work remotely varies."]})
    n = cc.QueryNormalizer.from_kb(kb)
    assert n.normalize("what is an sre").text == "what is an site reliability engineer"
    assert n.normalize("ml engineer pay").text == "machine learning engineer pay"
    assert n.normalize("learn html").text == "learn html"            # alias only on word boundaries
    assert n.normalize("what skils").text == "what skills"            # typo -> question word
    assert n.normalize("the weather").text == "the weather"           # never towards answer-only words
    assert n.normalize("What skills does a Site Reliability Engineer (SRE) need?").corrections == []
    assert n.unknown_words("how much does a pilot earn") == ["pilot"]


def test_decision_policy():
    bot = cc.CareerChatbot(thresholds=cc.Thresholds(min_similarity=0.4, min_role_conf=0.3, min_intent_conf=0.3))
    assert bot.decide(0.9, 0.9, 0.9) == "answer"
    assert bot.decide(0.1, 0.9, 0.9) == "clarify_role"
    assert bot.decide(0.9, 0.1, 0.9) == "clarify_intent"
    assert bot.decide(0.1, 0.1, 0.9) == "fallback"
    assert bot.decide(0.9, 0.9, 0.2) == "fallback"


def test_saved_model():
    if not cc.MODEL_PATH.exists():
        print("  (skipped: run the notebook first to create the model)")
        return
    bot = cc.CareerChatbot.load().warmup()
    assert len(bot.kb_) == 1620
    r = bot.respond("How much do Data Scientists earn?")
    assert (r.kind, r.role, r.intent) == ("answer", "Data Scientist", "salary") and r.text in set(bot.kb_.answer)
    assert bot.respond("what is the average salary?").kind == "clarify_role"
    assert bot.respond("tell me a joke").kind == "fallback"
    assert bot.respond("   ").kind == "invalid"
    assert bot.respond("x" * (cc.MAX_QUERY_CHARS + 1)).kind == "invalid"
    assert len(bot.suggest_followups("Data Scientist", exclude=["salary"])) == 3


def test_follow_up_memory():
    if not cc.MODEL_PATH.exists():
        return
    bot = cc.CareerChatbot.load()
    first = bot.respond("What does a Web Developer do?", cc.ConversationContext())
    follow = bot.respond("how much do they earn?", first.context)
    assert (follow.kind, follow.role, follow.intent, follow.used_memory) == ("answer", "Web Developer", "salary", "role")
    assert bot.respond("how much do they earn?").kind == "clarify_role"           # no context -> no memory
    assert bot.respond("tell me a joke", first.context).kind == "fallback"        # memory never answers off-topic
    other = bot.respond("what is an SRE?", first.context)                         # names another career
    assert not (other.kind == "answer" and other.role == "Web Developer")


def test_several_careers_in_one_question():
    if not cc.MODEL_PATH.exists():
        return
    bot = cc.CareerChatbot.load()
    r = bot.respond("What is a Data Analyst, Data Scientist, Business Analyst?", cc.ConversationContext())
    assert r.kind == "answer" and r.roles == ["Data Analyst", "Data Scientist", "Business Analyst"]
    assert all(f"**{role}:**" in r.text for role in r.roles)
    follow = bot.respond("how much do they earn?", r.context)                     # memory keeps all three
    assert follow.kind == "answer" and follow.intent == "salary" and follow.roles == r.roles
    eight = bot.respond("what is AI Researcher, Machine Learning Engineer, AI Software Engineer, Data Scientist, "
                        "Software Engineer, Web Developer ,Game Developer and Data Analyst")
    assert len(eight.roles) == 8                                                  # no silent truncation
    single = bot.respond("What skills and tools does a Data Analyst need?")       # one career, no split
    assert single.roles == [] and single.role == "Data Analyst"


def test_small_talk_and_career_discovery():
    if not cc.MODEL_PATH.exists():
        return
    bot = cc.CareerChatbot.load()
    if bot.meta_ is None:
        return
    assert bot.respond("hello").meta_intent == "greeting"
    assert bot.respond("thank you").meta_intent == "thanks"
    r = bot.respond("what is the best job for me?")
    assert r.meta_intent == "recommend" and not r.roles                          # asks for interests first
    r = bot.respond("i like coding and maths, which career suits me?", cc.ConversationContext())
    assert r.kind == "meta" and len(r.roles) == 3 and len(r.suggestions) == 3
    assert bot.respond("how much do they earn?", r.context).roles == r.roles      # follow-up on the suggestions
    r = bot.respond("list all the jobs for computer science")
    assert r.meta_intent == "list_careers" and "-" in r.text
    assert bot.respond("How much do Data Scientists earn?").kind == "answer"       # career questions unchanged
    assert bot.respond("what's the weather tomorrow").kind != "answer"


def test_encoder_round_trip_and_mismatch():
    """Invariant: a saved model names its encoder; loading it with a different one fails loudly."""
    if not cc.MODEL_PATH.exists():
        return
    bot = cc.CareerChatbot.load()
    bot.warmup()  # the encoder named by the model reproduces its stored embeddings
    if bot.uses_embeddings:
        wrong = "sentence-transformers/all-MiniLM-L6-v2" if bot.encoder != cc.EMBEDDING_MODEL else "BAAI/bge-small-en-v1.5"
        bot.encoder = wrong
        try:
            bot.warmup()
        except cc.EncoderMismatchError:
            pass
        else:
            raise AssertionError("a mismatched encoder was accepted")


def test_v1_model_still_loads():
    if V1_MODEL.exists():
        old = cc.CareerChatbot.load(V1_MODEL)
        assert old.respond("How much do Data Scientists earn?").kind == "answer"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
