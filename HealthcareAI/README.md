# HealthcareAI

A local AI demo project covering XGBoost, RAG, and Agents applied to healthcare use cases.
All models run locally via Ollama — no API keys required.

---

## Project Structure

```
HealthcareAI/
├── docs/
│   ├── cardiovascular_guidelines.txt
│   ├── diabetes_guidelines.txt
│   ├── hypertension_protocol.txt
│   └── medication_interactions.txt
├── chroma_db/                  ← auto-created on first RAG run
├── venv/                       ← auto-created during setup
├── xgboost_demo.py
├── rag_demo.py
├── agent_demo.py
├── requirements.txt
└── README.md
```

---

## Prerequisites

### 1. Python 3.10 or higher
Download from https://www.python.org/downloads/

### 2. Ollama
Download and install from https://ollama.com

After installing, pull the two required models by running these commands in your terminal:
```
ollama pull llama3.2
ollama pull nomic-embed-text
```

> **Note:** Ollama must be running in the background before you run the RAG or Agent demos.
> It starts automatically after installation, but if you restart your machine you may need
> to open the Ollama app again.

---

## Setup (one time only)

Open a terminal in your `HealthcareAI` folder and run the following commands in order.

**1. Create a virtual environment:**
```
python -m venv venv
```

**2. Activate the virtual environment:**
```
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal line.

**3. Install all dependencies:**
```
pip install -r requirements.txt
```

> You only need to do this setup once. For future sessions, just activate the venv
> with `venv\Scripts\activate` before running any demo.

---

## Running the Demos

Make sure your virtual environment is active (`(venv)` shown in terminal) before running any demo.

---

### Demo 1 — XGBoost
**What it covers:** Training a gradient boosting classifier on real breast cancer data,
evaluating with AUC-ROC, cross-validation, feature importance, and single patient prediction.

**No extra requirements** — uses sklearn's built-in dataset.

```
python xgboost_demo.py
```

---

### Demo 2 — RAG (Retrieval-Augmented Generation)
**What it covers:** Loading clinical guideline documents, chunking, embedding with
nomic-embed-text, storing in ChromaDB, and answering healthcare questions using
retrieved context vs. LLM memory alone.

**Requires:** Ollama running with `llama3.2` and `nomic-embed-text` pulled.

```
python rag_demo.py
```

> On first run this embeds all documents and saves them to `chroma_db/`.
> Subsequent runs load from disk and are faster.
>
> If you want to re-embed from scratch, delete the `chroma_db/` folder and run again.

---

### Demo 3 — Agent
**What it covers:** A ReAct agent that decides between two tools — RAG guidelines search
and XGBoost tumor prediction — based on the question asked. Demonstrates tool calling,
multi-tool sequencing, and agent reasoning.

**Requires:** Ollama running with `llama3.2` and `nomic-embed-text` pulled.
**Also requires:** RAG demo must have been run at least once so `chroma_db/` exists.

```
python agent_demo.py
```

> **Model note:** llama3.2 is a 3B parameter model. Tool calling works but can occasionally
> produce inconsistent final answers. For more reliable results, run:
> ```
> ollama pull llama3.1:8b
> ```
> Then open `agent_demo.py` and change line 31 to:
> ```python
> LLM_MODEL = "llama3.1:8b"
> ```

---

## Recommended Run Order

Run the demos in this order the first time:

```
1. python xgboost_demo.py
2. python rag_demo.py
3. python agent_demo.py
```

The agent demo depends on the `chroma_db/` folder created by the RAG demo,
so RAG must be run first.

---

## Potential Improvements

**XGBoost**
- Hyperparameter tuning with grid search or Optuna
- Handle imbalanced datasets with `scale_pos_weight`
- Save and reload the trained model with joblib instead of retraining every run
- Early stopping to prevent overfitting automatically

**RAG**
- Increase `k` to retrieve more chunks and improve answer completeness
- Hybrid search (semantic + keyword) for better retrieval accuracy
- Re-ranking retrieved chunks before passing to the LLM
- Swap in real clinical guidelines (ACC/AHA publish PDFs publicly)

**Agent**
- Upgrade to `llama3.1:8b` for more reliable tool calling
- Add more tools (e.g. drug dosage calculator, patient risk score)
- Add a max iterations limit to prevent infinite retry loops

**LangChain (not yet demonstrated)**
- LCEL chains as an explicit concept
- Prompt templates with variable injection
- Conversation memory so the agent remembers previous questions in a session
- Structured output parsing to get Python objects back instead of raw text

**General**
- A single `main.py` that runs all three demos in sequence
- A `config.py` to centralize model names, paths, and settings in one place
