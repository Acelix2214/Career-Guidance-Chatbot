"""Shared NLP components for the Career Guidance Support Chatbot.

Both ``career_guidance_chatbot.ipynb`` (training / evaluation) and ``app.py``
(Streamlit demo) import this module, so preprocessing, feature extraction and
response selection are identical in both places.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel

DATASET_ID = "Pradeep016/career-guidance-qa-dataset"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "career_chatbot.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
MAX_QUERY_CHARS = 500

# The dataset stores 30 questions per role: 10 topics x 3 paraphrases, always in
# the same order. The notebook verifies this before relying on it.
INTENTS = [
    "role_overview", "skills", "tools_technologies", "career_growth", "education",
    "daily_tasks", "salary", "work_environment", "challenges", "getting_started",
]
INTENT_NAMES = {
    "role_overview": "Role overview",
    "skills": "Skills",
    "tools_technologies": "Tools & technologies",
    "career_growth": "Career growth",
    "education": "Education & qualifications",
    "daily_tasks": "Daily tasks",
    "salary": "Salary",
    "work_environment": "Work environment & employers",
    "challenges": "Challenges",
    "getting_started": "Getting started",
}
PARAPHRASES_PER_INTENT = 3

# Question words carry intent ("where" -> work environment, "how much" -> salary),
# so they are kept even though sklearn's list treats them as stop words.
KEEP_WORDS = {"how", "much", "where", "what", "which", "why", "when", "do"}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS - KEEP_WORDS)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[+#]+|'[a-z]+)?")


def add_intent_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``intent`` and ``paraphrase_slot`` from each question's position in its role block."""
    out = df.copy()
    pos = out.groupby("role", sort=False).cumcount().to_numpy()
    out["intent"] = [INTENTS[p // PARAPHRASES_PER_INTENT] for p in pos]
    out["paraphrase_slot"] = pos % PARAPHRASES_PER_INTENT
    return out


class TextPreprocessor:
    """Normalise -> lowercase -> tokenise -> (optionally) drop stop words -> (optionally) stem.

    A class rather than a closure so it can be pickled inside the vectorizer.
    """

    def __init__(self, remove_stopwords: bool = True, stem: bool = True):
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self._stemmer = SnowballStemmer("english") if stem else None

    def tokens(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFKC", str(text)).lower().replace("’", "'")
        toks = [t.split("'")[0] for t in _TOKEN_RE.findall(text)]  # "what's" -> "what"
        if self.remove_stopwords:
            toks = [t for t in toks if t not in STOP_WORDS]
        if self._stemmer is not None:
            toks = [self._stemmer.stem(t) for t in toks]
        return toks

    def __call__(self, text: str) -> str:
        return " ".join(self.tokens(text))

    def __repr__(self) -> str:
        return f"TextPreprocessor(remove_stopwords={self.remove_stopwords}, stem={self.stem})"


def build_vectorizer(preprocessor: TextPreprocessor | None = None, ngram_range=(1, 2)) -> TfidfVectorizer:
    """TF-IDF over the preprocessed text (the preprocessor runs inside the vectorizer)."""
    return TfidfVectorizer(preprocessor=preprocessor or TextPreprocessor(),
                           token_pattern=r"\S+", ngram_range=ngram_range, sublinear_tf=True)


@lru_cache(maxsize=1)
def get_encoder(name: str = EMBEDDING_MODEL):
    """Sentence-embedding model (downloaded once from the Hugging Face Hub, then cached)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cpu")


def embed(texts, name: str = EMBEDDING_MODEL) -> np.ndarray:
    return get_encoder(name).encode(list(texts), normalize_embeddings=True, batch_size=64,
                                    show_progress_bar=False)


@dataclass
class Thresholds:
    min_similarity: float = 0.3  # best KB cosine similarity below this -> out of scope
    min_role_conf: float = 0.3   # role probability below this -> ask which career
    min_intent_conf: float = 0.3  # intent probability below this -> ask which topic


@dataclass
class ChatResponse:
    kind: str  # "answer" | "clarify_role" | "clarify_intent" | "fallback" | "invalid"
    text: str
    role: str | None = None
    intent: str | None = None
    role_conf: float = 0.0
    intent_conf: float = 0.0
    similarity: float = 0.0
    matched_question: str | None = None
    role_candidates: list[tuple[str, float]] = field(default_factory=list)
    intent_candidates: list[tuple[str, float]] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Joint confidence of the selected (role, intent) pair."""
        return self.role_conf * self.intent_conf


FALLBACK_TEXT = (
    "I'm not confident I can answer that. I'm a career-guidance assistant: try asking about a "
    "specific career's skills, salary, education, daily tasks, tools, challenges, work "
    "environment, career growth, or how to get started, e.g. *\"What skills does a Data Analyst need?\"*"
)


class CareerChatbot:
    """Hybrid NLP chatbot.

    * TF-IDF + Logistic Regression predicts the **career role** (54 classes).
    * Logistic Regression on TF-IDF *or* sentence embeddings predicts the **intent** (10 topics);
      ``augment_with_answers`` also trains it on the knowledge-base answers.
    * Cosine similarity picks the closest stored question inside the predicted (role, intent)
      group, and the best overall similarity is used for out-of-scope detection.
    * Confidence / similarity thresholds trigger clarification or fallback responses.
    """

    def __init__(self, intent_features: str = "embedding", augment_with_answers: bool = True,
                 role_C: float = 10.0, intent_C: float = 10.0, ngram_range=(1, 2),
                 thresholds: Thresholds | None = None, preprocessor: TextPreprocessor | None = None):
        if intent_features not in ("tfidf", "embedding"):
            raise ValueError("intent_features must be 'tfidf' or 'embedding'")
        self.intent_features = intent_features
        self.augment_with_answers = augment_with_answers
        self.role_C, self.intent_C, self.ngram_range = role_C, intent_C, ngram_range
        self.thresholds = thresholds or Thresholds()
        self.preprocessor = preprocessor or TextPreprocessor()

    @property
    def uses_embeddings(self) -> bool:
        return self.intent_features == "embedding"

    # ---------------------------------------------------------------- training
    def fit(self, kb: pd.DataFrame) -> "CareerChatbot":
        """``kb`` needs columns: role, intent, question, answer. Vectorizer is fitted on ``kb`` only."""
        self.kb_ = kb[["role", "intent", "question", "answer"]].reset_index(drop=True)
        q, a, y = self.kb_["question"], self.kb_["answer"], self.kb_["intent"]
        self.vectorizer_ = build_vectorizer(self.preprocessor, self.ngram_range)
        self.vectorizer_.fit(pd.concat([q, a]) if self.augment_with_answers else q)
        self.X_kb_ = self.vectorizer_.transform(q)
        self.role_clf_ = LogisticRegression(C=self.role_C, max_iter=5000).fit(self.X_kb_, self.kb_["role"])

        if self.uses_embeddings:
            self.E_kb_ = embed(q)
            X_int = np.vstack([self.E_kb_, embed(a)]) if self.augment_with_answers else self.E_kb_
        else:
            X_int = self.vectorizer_.transform(pd.concat([q, a])) if self.augment_with_answers else self.X_kb_
        y_int = pd.concat([y, y]) if self.augment_with_answers else y
        self.intent_clf_ = LogisticRegression(C=self.intent_C, max_iter=5000).fit(X_int, y_int)
        return self

    # --------------------------------------------------------------- inference
    def _features(self, queries):
        """Returns (TF-IDF matrix, intent-feature matrix, cosine similarity to every KB question)."""
        X = self.vectorizer_.transform(queries)
        if self.uses_embeddings:
            E = embed(queries)
            return X, E, E @ self.E_kb_.T
        return X, X, linear_kernel(X, self.X_kb_)

    def similarities(self, queries) -> np.ndarray:
        """Cosine similarity of each query to every KB question (the retrieval component)."""
        return self._features(list(queries))[2]

    def predict(self, queries) -> pd.DataFrame:
        """Vectorised (role, intent) prediction, used for evaluation."""
        X, Xi, sims = self._features(list(queries))
        pr, pi = self.role_clf_.predict_proba(X), self.intent_clf_.predict_proba(Xi)
        return pd.DataFrame({
            "pred_role": self.role_clf_.classes_[pr.argmax(1)], "role_conf": pr.max(1),
            "pred_intent": self.intent_clf_.classes_[pi.argmax(1)], "intent_conf": pi.max(1),
            "max_similarity": sims.max(1),
        })

    def decide(self, role_conf: float, intent_conf: float, max_similarity: float) -> str:
        """Response-selection policy shared by ``respond`` and the notebook's evaluation."""
        t = self.thresholds
        role_ok, intent_ok = role_conf >= t.min_role_conf, intent_conf >= t.min_intent_conf
        if max_similarity < t.min_similarity or not (role_ok or intent_ok):
            return "fallback"
        if not role_ok:
            return "clarify_role"
        if not intent_ok:
            return "clarify_intent"
        return "answer"

    def respond(self, query: str) -> ChatResponse:
        query = (query or "").strip()
        if not query:
            return ChatResponse("invalid", "Please type a question.")
        if len(query) > MAX_QUERY_CHARS:
            return ChatResponse("invalid", f"Please keep questions under {MAX_QUERY_CHARS} characters.")
        if not self.preprocessor.tokens(query):
            return ChatResponse("fallback", FALLBACK_TEXT)

        X, Xi, sims = self._features([query])
        pr, pi, sims = self.role_clf_.predict_proba(X)[0], self.intent_clf_.predict_proba(Xi)[0], sims[0]
        r_order, i_order = pr.argsort()[::-1], pi.argsort()[::-1]
        resp = ChatResponse(
            kind="fallback", text=FALLBACK_TEXT,
            role=self.role_clf_.classes_[r_order[0]], role_conf=float(pr[r_order[0]]),
            intent=self.intent_clf_.classes_[i_order[0]], intent_conf=float(pi[i_order[0]]),
            similarity=float(sims.max()),
            role_candidates=[(self.role_clf_.classes_[j], float(pr[j])) for j in r_order[:3]],
            intent_candidates=[(self.intent_clf_.classes_[j], float(pi[j])) for j in i_order[:3]],
        )
        resp.kind = self.decide(resp.role_conf, resp.intent_conf, resp.similarity)
        if resp.kind == "clarify_role":
            guesses = ", ".join(r for r, _ in resp.role_candidates)
            resp.text = (f"It sounds like you're asking about **{INTENT_NAMES[resp.intent].lower()}**, but I'm "
                         f"not sure which career you mean. Could you name the role? (Best guesses: {guesses}.)")
        elif resp.kind == "clarify_intent":
            topics = ", ".join(INTENT_NAMES[i].lower() for i in INTENTS)
            resp.text = (f"I think you're asking about **{resp.role}**, but I'm not sure what you'd like to "
                         f"know. I can tell you about: {topics}.")
        elif resp.kind == "answer":
            # Response selection: closest stored question inside the predicted (role, intent) group.
            kb = self.kb_
            idx = np.flatnonzero((kb["role"].to_numpy() == resp.role) & (kb["intent"].to_numpy() == resp.intent))
            best = idx[sims[idx].argmax()]
            resp.text, resp.matched_question = kb.at[best, "answer"], kb.at[best, "question"]
        return resp

    # ------------------------------------------------------------- persistence
    def save(self, path: Path = MODEL_PATH, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)  # the encoder itself is not pickled; it is re-loaded by name
        meta = {"sklearn_version": sklearn.__version__, "dataset": DATASET_ID,
                "embedding_model": EMBEDDING_MODEL if self.uses_embeddings else None,
                "intent_features": self.intent_features, "augment_with_answers": self.augment_with_answers,
                "kb_size": int(len(self.kb_)), "n_roles": int(self.kb_["role"].nunique()),
                "role_C": self.role_C, "intent_C": self.intent_C, "ngram_range": list(self.ngram_range),
                "thresholds": asdict(self.thresholds), **(metadata or {})}
        path.with_name(METADATA_PATH.name).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "CareerChatbot":
        # joblib uses pickle: only load model files produced by this project's notebook.
        model = joblib.load(Path(path))
        if not isinstance(model, CareerChatbot):
            raise TypeError(f"{path} does not contain a CareerChatbot")
        return model
