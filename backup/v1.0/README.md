# Career Guidance Support Chatbot

An NLP chatbot that answers career-guidance questions (skills, salary, education, daily tasks, tools,
challenges, work environment, career growth, getting started) for 54 careers. It combines text
classification and cosine-similarity retrieval, and uses confidence thresholds to ask for clarification or
fall back instead of guessing.

- **Main deliverable:** [`career_guidance_chatbot.ipynb`](career_guidance_chatbot.ipynb), the full pipeline with
  data analysis, training, evaluation, sample conversations and error analysis.
- **Demo:** [`app.py`](app.py), a Streamlit chat interface that loads the model saved by the notebook.

## Dataset

[`Pradeep016/career-guidance-qa-dataset`](https://huggingface.co/datasets/Pradeep016/career-guidance-qa-dataset)
on the Hugging Face Hub. It is downloaded automatically by `datasets.load_dataset` on the first run and
cached in `~/.cache/huggingface`.

- 1,620 rows; columns `role`, `question`, `answer`; no missing values or duplicates.
- 54 roles × 30 questions. Each role's questions cover **10 topics × 3 paraphrases** in a fixed order. The
  notebook verifies this structure (question templates and answer similarity) and derives an `intent` label from it.
- The answers (including salary figures) are used as provided by the dataset and are not independently
  verified. Check the dataset page for its license and terms before reusing it.

## Project structure

| Path | Purpose |
|---|---|
| `career_guidance_chatbot.ipynb` | Primary deliverable: the 17-section notebook (EDA → preprocessing → models → evaluation → analysis) |
| `chatbot_core.py` | Shared code: preprocessing, `CareerChatbot` model, response-selection policy, save/load. Imported by both the notebook and the app |
| `app.py` | Streamlit chat interface |
| `models/career_chatbot.joblib` | Trained model written by the notebook (vectorizer, classifiers, knowledge base, thresholds) |
| `models/model_metadata.json` | Model settings and the measured test / cross-validation metrics |
| `test_chatbot_core.py` | Self-checks for preprocessing, the decision policy and the saved model |
| `requirements.txt` | Python dependencies |
| `.venv/` | Project-local virtual environment (not required if you create your own) |

## Setup

Requires Python 3.10 or newer (developed with Python 3.14 on Windows 11, CPU only).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Optional: the CPU-only PyTorch build is a much smaller download
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The first run needs internet access to download the dataset and the
`sentence-transformers/all-MiniLM-L6-v2` model (about 90 MB). After that, both are cached locally.

## Run the notebook

```bash
jupyter notebook career_guidance_chatbot.ipynb
```

Select the `.venv` interpreter/kernel and choose **Run All**. A full run took about 1.5 minutes on a laptop CPU
once the dataset and MiniLM model were cached. The first run takes longer because it downloads them.
The notebook trains and evaluates the models and writes `models/career_chatbot.joblib`.

To run it non-interactively instead:

```bash
jupyter nbconvert --to notebook --execute --inplace career_guidance_chatbot.ipynb
```

## Launch the Streamlit app

Run the notebook first (it creates the model file), then:

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). The app shows a chat-style conversation with
session history and a confidence bar for each answer. A "How the model decided" panel lists the predicted
role and topic with their probabilities and the matched stored question. Sidebar buttons give example
questions and clear the conversation.

The app does **not** retrain anything. It loads the saved model once (`st.cache_resource`) and calls the same
`CareerChatbot.respond()` method that the notebook evaluates.

To check the shared code and the saved model:

```bash
python test_chatbot_core.py
```

## NLP pipeline and model

1. **Preprocessing** (`TextPreprocessor`): Unicode normalisation, lower-casing, regex tokenisation, and
   stop-word removal that *keeps* intent-bearing question words (`how`, `much`, `where`, …). Optional
   Snowball stemming. The variant is chosen by validation.
2. **Features:** TF-IDF (unigrams and bigrams, sub-linear TF) and pretrained MiniLM sentence embeddings.
3. **Models:**
   - **Role classifier:** TF-IDF + Logistic Regression (54 classes).
   - **Intent classifier:** Logistic Regression on MiniLM embeddings (10 classes). The notebook selected this
     over TF-IDF, Linear SVM and Complement Naive Bayes using validation on the training split only.
   - **Answer selection:** cosine similarity picks the closest stored question inside the predicted
     (role, intent) group and returns its answer.
4. **Response selection:** thresholds learned from training-side data decide between *answer*,
   *clarify role*, *clarify topic* and *fallback*. They are: best knowledge-base similarity
   (catches out-of-scope questions), role probability (catches questions with no career named), and intent
   probability (chosen to maximise correct minus wrong answers).

**Evaluation design.** The test set is a *leave-one-paraphrase-out* split. One wording from every
(role, topic) group is held out (540 questions), so every test question uses a wording never seen in training.
A sibling answer still exists in the knowledge base. Model selection uses an inner split of the training data,
and a 3-fold rotation of the held-out wording gives a cross-validated estimate.

## Results

All numbers below come from the executed notebook (see Sections 13–15 for the full tables).

| Metric (held-out test, unseen wordings) | Value |
|---|---|
| Role accuracy | 1.000 |
| Intent accuracy / macro-F1 | 0.715 / 0.646 |
| Retrieval-only Hit@1: TF-IDF / MiniLM | 0.293 / 0.700 |
| Hybrid model Hit@1 (test / 3-fold CV mean ± std) | 0.715 / 0.829 ± 0.140 |
| Coverage / accuracy of answered (with thresholds) | 0.987 / 0.720 |
| Fallback on 15 hand-written out-of-domain questions | 15 / 15 |
| Clarification on 6 hand-written role-free questions | 6 / 6 |
| 25 hand-written realistic questions answered correctly | 23 / 25 |

**Metrics used:**
- Classification is measured with accuracy and macro precision, recall and F1, plus a confusion matrix.
- Retrieval is measured with Hit@k and mean reciprocal rank (MRR).
- The full chatbot is measured with coverage, accuracy of answered questions, wrong-answer rate, and
  fallback/clarification rates.

## Limitations

- The dataset is small and templated: only 3 wordings per topic, one fixed answer per question, and only the
  54 careers it contains.
- Most errors come from topics whose wordings overlap. "Which programming languages…" (Skills) is predicted as
  Tools, and "educational requirements" is predicted as Getting started. The notebook shows these questions'
  answers overlap too. These errors are made *confidently*, so thresholds do not catch them.
- Role recognition needs the role name. Abbreviations not in the data ("ml engineer", "SRE") lead to a
  clarification request.
- The hand-written out-of-domain, role-free and probe sets are small and written for this project. They are
  indicative, not a benchmark.
- There is no conversation memory: each message is answered independently.

Possible improvements (paraphrase augmentation, encoder fine-tuning, role aliases, conversation state) are
discussed in Section 15 of the notebook.
