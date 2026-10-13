"""Shared NLP components for the Career Guidance Support Chatbot.

Both ``career_guidance_chatbot.ipynb`` (training / evaluation) and ``app.py``
(Streamlit demo) import this module, so query normalisation, preprocessing,
feature extraction and response selection are identical in both places.

Pipeline for one user message::

    raw text -> QueryNormalizer (aliases, typo correction)
             -> TextPreprocessor + TF-IDF ---------> role classifier (LogReg)
             -> sentence encoder (embeddings) -----> intent classifier (LogReg)
             -> cosine similarity to stored questions -> response selection
                (thresholds -> answer / clarify / fallback, conversation memory)
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field, replace
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
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # v1.0 encoder; a saved model names its own
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "career_chatbot.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
NLTK_DIR = PROJECT_DIR / "nltk_data"
MAX_QUERY_CHARS = 500
MEMORY_TURNS = 4  # a remembered career expires after this many turns without a new one
MAX_ROLES_PER_QUESTION = 10  # "What is a Data Analyst, Data Scientist, Business Analyst?" -> up to 10 answers

import nltk  # noqa: E402  (WordNet data, if downloaded for lemmatisation, lives in ./nltk_data)
if str(NLTK_DIR) not in nltk.data.path:
    nltk.data.path.append(str(NLTK_DIR))

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

# Conversation-level intents outside the QA dataset, recognised by embedding similarity to the
# example phrases in data/meta_intents.csv (see CareerChatbot._meta_response).
META_TEXT = {
    "greeting": ("Hi! I'm a career guidance assistant for 54 careers. Ask me about a career's skills, salary, "
                 "education, daily tasks, tools, challenges, work environment, career growth or how to get started, "
                 "or tell me what you enjoy (e.g. *\"I like coding and maths\"*) and I'll suggest careers."),
    "thanks": "You're welcome! Ask me anything else about careers.",
    "help": ("I can answer questions about 54 careers: what the job is, skills, tools, education, daily tasks, "
             "salary, work environment, challenges, career growth and how to get started. Try *\"What skills does "
             "a Data Analyst need?\"*, *\"ml engineer salary\"*, or tell me your interests and I'll suggest careers."),
    "ask_interests": ("Happy to help you choose! Tell me what you enjoy or are good at, e.g. *\"I like coding and "
                      "maths\"* or *\"I enjoy design and talking to people\"*, and I'll suggest careers that fit."),
}
# words that say *what kind of request* it is rather than *which interest or field* it is about
META_GENERIC = frozenset("""best good great right job jobs career careers work suit suits fit fits me my i choose pick
picking recommend recommendation suggest suggestion which what should know list show give tell all kind kinds type types
available related cover field fields area areas like likes enjoy enjoys love loves want someone people person get can
could good really something option options future path become do does""".split())

# Question words carry intent ("where" -> work environment, "how much" -> salary),
# so they are kept even though sklearn's list treats them as stop words.
KEEP_WORDS = {"how", "much", "where", "what", "which", "why", "when", "do"}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS - KEEP_WORDS)

# Hand-checked abbreviations -> words used in the dataset's role names. Acronyms that
# appear in role names, e.g. "(SRE)", are added automatically by QueryNormalizer.
CURATED_ALIASES = {
    "ml": "machine learning",
    "swe": "software engineer",
    "sde": "software engineer",
    "sysadmin": "systems administrator",
    "bi": "business intelligence",
    "dev": "developer",
    "devs": "developers",
    "viz": "visualization",
    "admin": "administrator",
    "mgr": "manager",
    "ecommerce": "e-commerce",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[+#]+|'[a-z]+)?")
_WORD_RE = re.compile(r"[A-Za-z]+")
# separators between careers in one question (not "/", which is part of "UX/UI Designer")
_LIST_SPLIT_RE = re.compile(r"\s*(?:,|;|&|\band\b|\bor\b|\bvs\.?|\bversus\b)\s*", re.IGNORECASE)


def add_intent_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``intent`` and ``paraphrase_slot`` from each question's position in its role block."""
    out = df.copy()
    pos = out.groupby("role", sort=False).cumcount().to_numpy()
    out["intent"] = [INTENTS[p // PARAPHRASES_PER_INTENT] for p in pos]
    out["paraphrase_slot"] = pos % PARAPHRASES_PER_INTENT
    return out


def role_base_name(role: str) -> str:
    """'Product Owner (Entry-level)' -> 'Product Owner' (the way questions usually name a role)."""
    return re.sub(r"\s*\(.*?\)", "", role).strip()


# ----------------------------------------------------------------------------- preprocessing
def _wordnet_lemmatizer():
    """WordNet lemmatizer; downloads the WordNet data into ./nltk_data on first use."""
    import nltk
    from nltk.stem import WordNetLemmatizer

    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.append(str(NLTK_DIR))
    lemmatizer = WordNetLemmatizer()
    try:
        lemmatizer.lemmatize("tests")
    except LookupError:
        nltk.download("wordnet", download_dir=str(NLTK_DIR), quiet=True)
        lemmatizer.lemmatize("tests")
    return lemmatizer


class TextPreprocessor:
    """Normalise -> lowercase -> tokenise -> (optionally) drop stop words -> stem or lemmatise.

    A class rather than a closure so it can be pickled inside the vectorizer.
    """

    lemmatize = False  # class default keeps models pickled before this option loadable
    _lemmatizer = None

    def __init__(self, remove_stopwords: bool = True, stem: bool = True, lemmatize: bool = False):
        if stem and lemmatize:
            raise ValueError("choose stemming or lemmatisation, not both")
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.lemmatize = lemmatize
        self._stemmer = SnowballStemmer("english") if stem else None
        self._lemmatizer = _wordnet_lemmatizer() if lemmatize else None

    def tokens(self, text: str) -> list[str]:
        """Return the processed tokens of ``text``."""
        text = unicodedata.normalize("NFKC", str(text)).lower().replace("’", "'")
        toks = [t.split("'")[0] for t in _TOKEN_RE.findall(text)]  # "what's" -> "what"
        if self.remove_stopwords:
            toks = [t for t in toks if t not in STOP_WORDS]
        if self._stemmer is not None:
            toks = [self._stemmer.stem(t) for t in toks]
        elif self._lemmatizer is not None:
            toks = [self._lemmatizer.lemmatize(self._lemmatizer.lemmatize(t, "n"), "v") for t in toks]
        return toks

    def __call__(self, text: str) -> str:
        return " ".join(self.tokens(text))

    def __repr__(self) -> str:
        return (f"TextPreprocessor(remove_stopwords={self.remove_stopwords}, stem={self.stem}, "
                f"lemmatize={self.lemmatize})")


def build_vectorizer(preprocessor: TextPreprocessor | None = None, ngram_range=(1, 2),
                     analyzer: str = "word") -> TfidfVectorizer:
    """TF-IDF over the preprocessed text. ``analyzer='char_wb'`` builds character n-grams instead."""
    if analyzer == "word":
        return TfidfVectorizer(preprocessor=preprocessor or TextPreprocessor(),
                               token_pattern=r"\S+", ngram_range=ngram_range, sublinear_tf=True)
    return TfidfVectorizer(preprocessor=preprocessor or TextPreprocessor(), analyzer=analyzer,
                           ngram_range=ngram_range, sublinear_tf=True)


# ----------------------------------------------------------------------------- query normalisation
@dataclass
class NormalizedQuery:
    text: str
    corrections: list[tuple[str, str]] = field(default_factory=list)  # (what the user typed, what we read)


class QueryNormalizer:
    """Rewrites a user query before feature extraction: role aliases first, then typo correction.

    Typos are corrected only towards words of the knowledge base it was fitted on, so an
    off-topic word ("weather") is never turned into a career word.
    """

    def __init__(self, aliases: dict[str, str], vocabulary: set[str], targets: set[str],
                 cutoff: float = 0.84, min_len: int = 4):
        self.aliases = aliases
        self.vocabulary = vocabulary  # every known word: never "corrected"
        self.cutoff, self.min_len = cutoff, min_len
        self._targets = sorted(w for w in targets if len(w) >= 3)  # words a typo may be corrected to
        keys = sorted(aliases, key=len, reverse=True)  # longest alias wins
        self._alias_re = re.compile(r"\b(" + "|".join(map(re.escape, keys)) + r")\b", re.IGNORECASE) if keys else None

    @classmethod
    def from_kb(cls, kb: pd.DataFrame) -> "QueryNormalizer":
        """Known words = KB questions, answers and role names; typo targets = question and role words only."""
        words = lambda texts: {w.lower() for t in texts for w in _WORD_RE.findall(t)}
        question_words = words(kb["question"])
        targets = question_words | words(kb["role"])
        vocab = targets | words(kb["answer"])
        aliases = dict(CURATED_ALIASES)
        for role in kb["role"].unique():
            for m in re.finditer(r"\(([A-Z]{2,5})\)", role):  # "Site Reliability Engineer (SRE)"
                parts = role[: m.start()].split()[-len(m.group(1)):]
                if "".join(w[0] for w in parts).upper() == m.group(1):
                    aliases[m.group(1).lower()] = " ".join(parts).lower()
        # never rewrite a word the stored questions already use (e.g. "CSR Manager"): the model knows it
        aliases = {k: v for k, v in aliases.items() if k not in question_words}
        return cls(aliases, vocab, targets)

    def normalize(self, text: str) -> NormalizedQuery:
        """Return the rewritten query and the list of (original, replacement) changes."""
        corrections: list[tuple[str, str]] = []

        def _alias(m):
            new = self.aliases[m.group(0).lower()]
            if new in text.lower():  # "Site Reliability Engineer (SRE)": the full name is already there
                return m.group(0)
            corrections.append((m.group(0), new))
            return new

        out = self._alias_re.sub(_alias, text) if self._alias_re else text

        def _typo(m):
            word = m.group(0)
            low = word.lower()
            if len(low) < self.min_len or low in self.vocabulary or low in STOP_WORDS:
                return word
            # typos rarely change the first letter; requiring it avoids "learn" -> "earn"
            best = difflib.get_close_matches(low, [w for w in self._targets if w[0] == low[0]], n=1,
                                             cutoff=self.cutoff)
            if not best:
                return word
            corrections.append((word, best[0]))
            return best[0]

        return NormalizedQuery(_WORD_RE.sub(_typo, out), corrections)

    def unknown_words(self, text: str) -> list[str]:
        """Content words not in the KB vocabulary (e.g. an unseen job title such as 'pilot')."""
        return [w for w in (x.lower() for x in _WORD_RE.findall(text))
                if len(w) >= 3 and w not in STOP_WORDS and w not in self.vocabulary]


# ----------------------------------------------------------------------------- sentence encoder
@lru_cache(maxsize=4)
def get_encoder(name: str = EMBEDDING_MODEL):
    """Sentence-embedding model, loaded from the local cache when possible (faster, works offline)."""
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(name, device="cpu", local_files_only=True)
    except Exception:  # not cached yet -> download once from the Hugging Face Hub
        return SentenceTransformer(name, device="cpu")


def embed(texts, name: str = EMBEDDING_MODEL, prompt: str = "") -> np.ndarray:
    """L2-normalised embeddings of ``texts`` (``prompt`` is prepended, e.g. 'query: ' for e5 models)."""
    return get_encoder(name).encode([prompt + t for t in texts], normalize_embeddings=True,
                                    batch_size=64, show_progress_bar=False)


class EncoderMismatchError(RuntimeError):
    """The loaded sentence encoder does not reproduce the embeddings the model was trained with."""


# ----------------------------------------------------------------------------- response objects
@dataclass
class Thresholds:
    min_similarity: float = 0.3  # best KB cosine similarity below this -> out of scope
    min_role_conf: float = 0.3   # role probability below this -> ask which career
    min_intent_conf: float = 0.3  # intent probability below this -> ask which topic
    max_ambiguity_margin: float = 0.0  # top-1 minus top-2 intent prob below this -> offer "Did you mean"
    min_meta_similarity: float = 1.01  # similarity to a small-talk / discovery example needed (>1 = off)
    min_meta_margin: float = 0.0  # ...and how much it must beat the career questions to replace a clarification


@dataclass(frozen=True)
class ConversationContext:
    """What the bot remembers between turns of one conversation (kept by the caller, never on the bot)."""
    last_role: str | None = None
    last_intent: str | None = None
    turns_since_role: int = 0
    last_roles: tuple[str, ...] = ()  # when the last answer covered several careers


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
    normalized_query: str | None = None
    corrections: list[tuple[str, str]] = field(default_factory=list)
    alternative: tuple[str, str] | None = None  # (intent, stored question) offered as "Did you mean"
    used_memory: str | None = None  # "role" or "intent" when the previous turn filled a gap
    roles: list[str] = field(default_factory=list)  # all careers answered when the question named several
    meta_intent: str | None = None  # greeting / thanks / help / recommend / list_careers
    suggestions: list[str] = field(default_factory=list)  # follow-up questions offered by meta responses
    context: ConversationContext | None = None

    @property
    def confidence(self) -> float:
        """Joint confidence of the selected (role, intent) pair."""
        return self.role_conf * self.intent_conf


FALLBACK_TEXT = (
    "I'm not confident I can answer that. I'm a career-guidance assistant: try asking about a "
    "specific career's skills, salary, education, daily tasks, tools, challenges, work "
    "environment, career growth, or how to get started, e.g. *\"What skills does a Data Analyst need?\"*"
)


# ----------------------------------------------------------------------------- the chatbot
class CareerChatbot:
    """Hybrid NLP chatbot.

    * ``QueryNormalizer`` expands role abbreviations and fixes typos in the user query.
    * TF-IDF + Logistic Regression predicts the **career role** (54 classes).
    * Logistic Regression on sentence embeddings (or TF-IDF) predicts the **intent** (10 topics).
    * Cosine similarity picks the closest stored question inside the predicted (role, intent)
      group; the best similarity of the *raw* query decides whether the question is in scope.
    * Thresholds trigger clarification / fallback; ``ConversationContext`` fills a missing
      career or topic from the previous turn.
    """

    # class defaults keep models pickled by older versions loadable
    encoder = EMBEDDING_MODEL
    encoder_prompt = ""
    use_normalizer = True
    gate_on_raw = True
    normalizer_ = None
    n_extra_ = 0
    meta_ = None

    def __init__(self, intent_features: str = "embedding", augment_with_answers: bool = True,
                 role_C: float = 10.0, intent_C: float = 10.0, ngram_range=(1, 2),
                 thresholds: Thresholds | None = None, preprocessor: TextPreprocessor | None = None,
                 encoder: str = EMBEDDING_MODEL, encoder_prompt: str = "", use_normalizer: bool = True,
                 gate_on_raw: bool = True):
        if intent_features not in ("tfidf", "embedding"):
            raise ValueError("intent_features must be 'tfidf' or 'embedding'")
        self.intent_features = intent_features
        self.augment_with_answers = augment_with_answers
        self.role_C, self.intent_C, self.ngram_range = role_C, intent_C, ngram_range
        self.thresholds = thresholds or Thresholds()
        self.preprocessor = preprocessor or TextPreprocessor()
        self.encoder, self.encoder_prompt = encoder, encoder_prompt
        self.use_normalizer = use_normalizer
        self.gate_on_raw = gate_on_raw  # in-scope check on the raw query (True) or the normalised one

    @property
    def uses_embeddings(self) -> bool:
        return self.intent_features == "embedding"

    def _embed(self, texts) -> np.ndarray:
        return embed(texts, self.encoder, self.encoder_prompt)

    # ---------------------------------------------------------------- training
    def fit(self, kb: pd.DataFrame, extra: pd.DataFrame | None = None,
            meta: pd.DataFrame | None = None) -> "CareerChatbot":
        """``kb`` needs columns: role, intent, question, answer. Everything is fitted on ``kb`` only,
        plus optional ``extra`` training questions (role, intent, question) that are used by the two
        classifiers but never become answerable entries of the knowledge base."""
        self.kb_ = kb[["role", "intent", "question", "answer"]].reset_index(drop=True)
        q, a, y = self.kb_["question"], self.kb_["answer"], self.kb_["intent"]
        xq = extra["question"] if extra is not None else q.iloc[:0]
        xr = extra["role"] if extra is not None else self.kb_["role"].iloc[:0]
        xi = extra["intent"] if extra is not None else y.iloc[:0]
        self.n_extra_ = len(xq)
        self.normalizer_ = QueryNormalizer.from_kb(self.kb_)
        self.vectorizer_ = build_vectorizer(self.preprocessor, self.ngram_range)
        self.vectorizer_.fit(pd.concat([q, xq, a]) if self.augment_with_answers else pd.concat([q, xq]))
        self.X_kb_ = self.vectorizer_.transform(q)
        X_role = self.vectorizer_.transform(pd.concat([q, xq]))
        self.role_clf_ = LogisticRegression(C=self.role_C, max_iter=5000).fit(X_role, pd.concat([self.kb_["role"], xr]))

        if self.uses_embeddings:
            self.E_kb_ = self._embed(q)
            parts = [self.E_kb_] + ([self._embed(xq)] if len(xq) else []) + ([self._embed(a)] if self.augment_with_answers else [])
            X_int = np.vstack(parts)
        else:
            X_int = self.vectorizer_.transform(pd.concat([q, xq, a]) if self.augment_with_answers else pd.concat([q, xq]))
        y_int = pd.concat([y, xi, y]) if self.augment_with_answers else pd.concat([y, xi])
        self.intent_clf_ = LogisticRegression(C=self.intent_C, max_iter=5000).fit(X_int, y_int)
        if meta is not None and self.uses_embeddings:
            self._fit_meta(meta)
        return self

    def _fit_meta(self, meta: pd.DataFrame) -> None:
        """Embed the small-talk / discovery examples and build one profile vector per career
        (the mean embedding of its stored answers) for interest-based career suggestions."""
        self.meta_ = meta[["intent", "example"]].reset_index(drop=True)
        self.E_meta_ = self._embed(self.meta_["example"])
        E_ans = self._embed(self.kb_["answer"])
        self.role_names_ = np.array(sorted(self.kb_["role"].unique()))
        prof = np.vstack([E_ans[(self.kb_["role"] == r).to_numpy()].mean(0) for r in self.role_names_])
        self.role_profiles_ = prof / np.linalg.norm(prof, axis=1, keepdims=True)

    # --------------------------------------------------------------- inference
    def normalize(self, query: str) -> NormalizedQuery:
        """Apply the query normaliser (a no-op when disabled or for models saved before it existed)."""
        if self.use_normalizer and self.normalizer_ is not None:
            return self.normalizer_.normalize(query)
        return NormalizedQuery(query)

    def _features(self, queries, raw=None):
        """(TF-IDF matrix, intent features, cosine sims to every KB question, in-scope similarity).

        ``raw`` are the un-normalised queries; with ``gate_on_raw`` the in-scope check uses them so
        that a typo correction can never pull an off-topic question into the domain.
        """
        queries = list(queries)
        raw = queries if raw is None or not self.gate_on_raw else list(raw)
        X = self.vectorizer_.transform(queries)
        if self.uses_embeddings:
            E = self._embed(queries)
            sims = E @ self.E_kb_.T
            changed = [i for i, (a, b) in enumerate(zip(queries, raw)) if a != b]
            gate = sims.max(1).copy()
            if changed:
                gate[changed] = (self._embed([raw[i] for i in changed]) @ self.E_kb_.T).max(1)
            return X, E, sims, gate
        sims = linear_kernel(X, self.X_kb_)
        gate = linear_kernel(self.vectorizer_.transform(raw), self.X_kb_).max(1)
        return X, X, sims, gate

    def similarities(self, queries) -> np.ndarray:
        """Cosine similarity of each query to every KB question (the retrieval component)."""
        return self._features(list(queries))[2]

    def predict(self, queries, normalize: bool = True) -> pd.DataFrame:
        """Vectorised (role, intent) prediction used for evaluation."""
        raw = list(queries)
        norm = [self.normalize(q).text for q in raw] if normalize else raw
        X, Xi, _, gate = self._features(norm, raw)
        pr, pi = self.role_clf_.predict_proba(X), self.intent_clf_.predict_proba(Xi)
        order = np.argsort(-pi, axis=1)
        rows = np.arange(len(raw))
        return pd.DataFrame({
            "pred_role": self.role_clf_.classes_[pr.argmax(1)], "role_conf": pr.max(1),
            "pred_intent": self.intent_clf_.classes_[order[:, 0]], "intent_conf": pi[rows, order[:, 0]],
            "intent2": self.intent_clf_.classes_[order[:, 1]], "intent2_conf": pi[rows, order[:, 1]],
            "max_similarity": gate, "normalized": norm,
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

    def _best_in_group(self, role: str, intent: str, sims: np.ndarray) -> int:
        kb = self.kb_
        idx = np.flatnonzero((kb["role"].to_numpy() == role) & (kb["intent"].to_numpy() == intent))
        return int(idx[sims[idx].argmax()])

    def _respond_once(self, query: str) -> ChatResponse:
        nq = self.normalize(query)
        X, Xi, sims, gate = self._features([nq.text], [query])
        pr, pi, sims = self.role_clf_.predict_proba(X)[0], self.intent_clf_.predict_proba(Xi)[0], sims[0]
        r_order, i_order = pr.argsort()[::-1], pi.argsort()[::-1]
        resp = ChatResponse(
            kind="fallback", text=FALLBACK_TEXT,
            role=self.role_clf_.classes_[r_order[0]], role_conf=float(pr[r_order[0]]),
            intent=self.intent_clf_.classes_[i_order[0]], intent_conf=float(pi[i_order[0]]),
            similarity=float(gate[0]),
            role_candidates=[(self.role_clf_.classes_[j], float(pr[j])) for j in r_order[:3]],
            intent_candidates=[(self.intent_clf_.classes_[j], float(pi[j])) for j in i_order[:3]],
            normalized_query=nq.text, corrections=nq.corrections,
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
            self._fill_answer(resp, sims)
            intent2, p2 = resp.intent_candidates[1]
            if resp.intent_conf - p2 < self.thresholds.max_ambiguity_margin:
                resp.alternative = (intent2, self.kb_.at[self._best_in_group(resp.role, intent2, sims), "question"])
        resp._sims = sims  # kept for memory-based re-selection within this call only
        return resp

    def _fill_answer(self, resp: ChatResponse, sims: np.ndarray) -> None:
        best = self._best_in_group(resp.role, resp.intent, sims)
        resp.text, resp.matched_question = self.kb_.at[best, "answer"], self.kb_.at[best, "question"]

    def respond(self, query: str, context: ConversationContext | None = None) -> ChatResponse:
        """Answer one message. Pass the previous ``response.context`` to enable follow-up memory."""
        query = (query or "").strip()
        if not query:
            return ChatResponse("invalid", "Please type a question.", context=context)
        if len(query) > MAX_QUERY_CHARS:
            return ChatResponse("invalid", f"Please keep questions under {MAX_QUERY_CHARS} characters.",
                                context=context)
        if not self.preprocessor.tokens(query):
            return ChatResponse("fallback", FALLBACK_TEXT, context=context)

        resp = self._respond_once(query)
        ctx = context or ConversationContext()
        multi = self._multi_role_answer(query, resp)
        if multi is not None:
            resp = multi
        elif context is not None:
            resp = self._apply_memory(query, resp, ctx)
        if resp.kind in ("fallback", "clarify_role", "clarify_intent"):
            resp = self._meta_response(query, resp) or resp
        resp.__dict__.pop("_sims", None)

        if resp.kind == "meta":
            resp.context = (ConversationContext(resp.roles[0], None, 0, tuple(resp.roles)) if resp.roles
                            else replace(ctx, turns_since_role=ctx.turns_since_role + 1))
            return resp
        if resp.kind == "answer":
            resp.context = ConversationContext(resp.role, resp.intent, 0,
                                               tuple(resp.roles) if len(resp.roles) > 1 else ())
        elif resp.kind == "clarify_role":
            resp.context = replace(ctx, last_intent=resp.intent, turns_since_role=ctx.turns_since_role + 1)
        elif resp.kind == "clarify_intent":
            resp.context = ConversationContext(resp.role, ctx.last_intent, 0)
        else:
            resp.context = replace(ctx, turns_since_role=ctx.turns_since_role + 1)
        return resp

    def _meta_response(self, query: str, resp: ChatResponse) -> ChatResponse | None:
        """Greetings, thanks, help and career discovery ("best job for me", "jobs for computer science"):
        nearest example phrase by embedding similarity; only when it beats every stored career question."""
        if self.meta_ is None:
            return None
        text = resp.normalized_query or query
        e = self._embed([text])
        sims = (e @ self.E_meta_.T)[0]
        j = int(sims.argmax())
        margin = sims[j] - resp.similarity
        if sims[j] < self.thresholds.min_meta_similarity or margin <= 0:
            return None
        if resp.kind != "fallback" and margin < self.thresholds.min_meta_margin:
            return None  # a role-free career question keeps its "which career?" clarification
        intent = self.meta_.at[j, "intent"]
        out = ChatResponse(kind="meta", text=META_TEXT.get(intent, ""), meta_intent=intent,
                           similarity=float(sims[j]), normalized_query=resp.normalized_query,
                           corrections=resp.corrections)
        if intent in ("recommend", "list_careers"):
            # the interest / field words, e.g. "computer science" in "list jobs for computer science"
            words = [w for w in self.preprocessor.tokens(text) if w not in META_GENERIC and len(w) >= 3
                     and w in self.normalizer_.vocabulary]
            if not words and intent == "recommend":
                out.text = META_TEXT["ask_interests"]
                return out
            if not words:
                out.text = "Here are all the careers I know about:\n\n" + "\n".join(f"- {r}" for r in self.role_names_)
                return out
            prof = (self._embed([" ".join(words)]) @ self.role_profiles_.T)[0]
            k = 3 if intent == "recommend" else 8
            top = [self.role_names_[i] for i in np.argsort(-prof)[:k]]
            if intent == "recommend":
                lines = [f"{n}. **{r}** — {self._overview(r)}" for n, r in enumerate(top, 1)]
                out.text = ("Based on what you told me, these careers could fit:\n\n" + "\n".join(lines)
                            + "\n\nAsk me about any of them, e.g. their skills or salary.")
                out.roles = top
            else:
                out.text = (f"Careers I know that are closest to *{' '.join(words)}*:\n\n"
                            + "\n".join(f"- {r}" for r in top))
            out.suggestions = [f"What does a {role_base_name(r)} do?" for r in top[:3]]
        return out

    def _overview(self, role: str) -> str:
        """First sentence of the career's stored role-overview answer."""
        kb = self.kb_
        a = kb[(kb["role"].to_numpy() == role) & (kb["intent"].to_numpy() == "role_overview")]["answer"].iloc[0]
        return a.split(". ")[0].rstrip(".") + "."

    def _multi_role_answer(self, query: str, resp: ChatResponse) -> ChatResponse | None:
        """A question naming several careers ("What is a Data Analyst, Data Scientist, Business Analyst?"):
        run the role classifier on each listed part and answer the (shared) topic for every confident career."""
        t = self.thresholds
        if resp.similarity < t.min_similarity or resp.intent_conf < t.min_intent_conf:
            return None
        parts = [p for p in _LIST_SPLIT_RE.split(resp.normalized_query or query) if p and p.strip()]
        if len(parts) < 2:
            return None
        roles, confs = [], []
        for part in parts:
            pr = self.role_clf_.predict_proba(self.vectorizer_.transform([part]))[0]
            j = int(pr.argmax())
            if pr[j] >= t.min_role_conf and self.role_clf_.classes_[j] not in roles:
                roles.append(self.role_clf_.classes_[j]); confs.append(float(pr[j]))
        if len(roles) < 2:
            return None
        skipped = roles[MAX_ROLES_PER_QUESTION:]
        roles = roles[:MAX_ROLES_PER_QUESTION]
        answers = [self.kb_.at[self._best_in_group(r, resp.intent, resp._sims), "answer"] for r in roles]
        resp.kind, resp.roles, resp.role, resp.role_conf = "answer", roles, roles[0], min(confs[:len(roles)])
        resp.text = "\n\n".join(f"**{r}:** {a}" for r, a in zip(roles, answers))
        if skipped:
            resp.text += f"\n\n(I answered the first {len(roles)} careers; ask again about: {', '.join(skipped)}.)"
        resp.matched_question = self.kb_.at[self._best_in_group(roles[0], resp.intent, resp._sims), "question"]
        resp.alternative = None
        return resp

    def _apply_memory(self, query: str, resp: ChatResponse, ctx: ConversationContext) -> ChatResponse:
        """Fill a missing career (or topic) from the previous turn — only when it is safe to."""
        remembered = ctx.last_roles or ((ctx.last_role,) if ctx.last_role else ())
        if (resp.kind == "clarify_role" and remembered and ctx.turns_since_role < MEMORY_TURNS
                and self.normalizer_ is not None
                and not self.normalizer_.unknown_words(resp.normalized_query or query)
                and not self._names_other_role(resp.normalized_query or query, remembered)):
            # re-run the model with each remembered career named explicitly
            agains = [self._respond_once(f"{query} ({role_base_name(r)})") for r in remembered]
            if all(a.kind == "answer" and a.role == r for a, r in zip(agains, remembered)):
                again = agains[0]
                if len(remembered) > 1:
                    again.roles = list(remembered)
                    again.text = "\n\n".join(f"**{r}:** {a.text}" for r, a in zip(remembered, agains))
                    again.role_conf = min(a.role_conf for a in agains)
                again.used_memory = "role"
                return again
        if resp.kind == "clarify_intent" and ctx.last_intent:
            resp.kind, resp.intent, resp.used_memory = "answer", ctx.last_intent, "intent"
            self._fill_answer(resp, resp._sims)
        return resp

    def _names_other_role(self, text: str, last_roles) -> bool:
        """True if ``text`` contains a word of another career's name (e.g. 'sre', 'analyst') that is not
        part of the remembered career(s) — then the user means a different career, so memory must not apply."""
        last_roles = (last_roles,) if isinstance(last_roles, str) else last_roles
        last = {w.lower() for r in last_roles for w in _WORD_RE.findall(r)}
        role_words = {w.lower() for r in self.kb_["role"].unique() for w in _WORD_RE.findall(r)}
        generic = {"junior", "level", "entry", "and"}
        return any(w in role_words and w not in last and w not in generic
                   for w in (x.lower() for x in _WORD_RE.findall(text)))

    def suggest_followups(self, role: str, exclude=(), k: int = 3) -> list[str]:
        """Up to ``k`` stored questions about ``role`` on topics not in ``exclude`` (for suggestion chips)."""
        kb = self.kb_[self.kb_["role"].to_numpy() == role]
        firsts = kb.drop_duplicates("intent").set_index("intent")["question"]
        return [firsts[i] for i in INTENTS if i in firsts.index and i not in set(exclude)][:k]

    # ------------------------------------------------------------- persistence
    def warmup(self) -> "CareerChatbot":
        """Load the sentence encoder and check it reproduces the stored embeddings (fail loudly if not)."""
        if self.uses_embeddings:
            probe = self._embed(self.kb_["question"][:3])
            cos = float((probe * self.E_kb_[:3]).sum(1).min())
            if cos < 0.999:
                raise EncoderMismatchError(
                    f"Encoder '{self.encoder}' does not reproduce the model's stored embeddings "
                    f"(cosine {cos:.3f}). Re-run the notebook to rebuild the model.")
        self.respond("What does a Data Analyst do?")
        return self

    def save(self, path: Path = MODEL_PATH, metadata: dict | None = None) -> None:
        """Pickle the model with joblib and write a JSON metadata file next to it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)  # the encoder itself is not pickled; it is re-loaded by name
        meta = {"sklearn_version": sklearn.__version__, "dataset": DATASET_ID,
                "embedding_model": self.encoder if self.uses_embeddings else None,
                "encoder_prompt": self.encoder_prompt,
                "intent_features": self.intent_features, "augment_with_answers": self.augment_with_answers,
                "kb_size": int(len(self.kb_)), "n_roles": int(self.kb_["role"].nunique()),
                "role_C": self.role_C, "intent_C": self.intent_C, "ngram_range": list(self.ngram_range),
                "use_normalizer": self.use_normalizer, "gate_on_raw": self.gate_on_raw,
                "extra_training_questions": int(self.n_extra_),
                "thresholds": asdict(self.thresholds), **(metadata or {})}
        path.with_name(METADATA_PATH.name).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "CareerChatbot":
        """Load a model saved by ``save``. joblib uses pickle: only load files made by this project."""
        model = joblib.load(Path(path))
        if not isinstance(model, CareerChatbot):
            raise TypeError(f"{path} does not contain a CareerChatbot")
        return model
