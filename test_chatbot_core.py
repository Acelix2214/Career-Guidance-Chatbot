"""Self-checks for chatbot_core and the saved model. Run: python test_chatbot_core.py"""
import pandas as pd

import chatbot_core as cc


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
    bot = cc.CareerChatbot.load()
    assert len(bot.kb_) == 1620
    r = bot.respond("How much do Data Scientists earn?")
    assert (r.kind, r.role, r.intent) == ("answer", "Data Scientist", "salary") and r.text in set(bot.kb_.answer)
    assert bot.respond("what is the average salary?").kind == "clarify_role"
    assert bot.respond("tell me a joke").kind == "fallback"
    assert bot.respond("   ").kind == "invalid"
    assert bot.respond("x" * (cc.MAX_QUERY_CHARS + 1)).kind == "invalid"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
