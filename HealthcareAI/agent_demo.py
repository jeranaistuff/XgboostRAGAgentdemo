# agent_demo.py
# Demonstrates a simple ReAct agent that decides between two tools:
#
#   Tool 1: search_guidelines  — RAG search over clinical documents
#   Tool 2: predict_tumor      — XGBoost breast cancer classifier
#
# The agent receives a question, reasons about which tool to use,
# calls it, and synthesizes a final answer. You'll see its thinking.
#
# NOTE: Uses ChatOllama (not OllamaLLM) — agents require a chat model
#       because they need to handle tool call messages back and forth.
#
# MODEL NOTE: llama3.2 (3B) is a small model — tool calling can be
#             unreliable. For better results: ollama pull llama3.1:8b
#             and set LLM_MODEL = "llama3.1:8b" below.

import json
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.prebuilt import create_react_agent

warnings.filterwarnings("ignore", category=DeprecationWarning)

LLM_MODEL   = "llama3.2"   # swap to "llama3.1:8b" for more reliable tool calling
EMBED_MODEL = "nomic-embed-text"
DB_DIR      = "chroma_db"


# ─────────────────────────────────────────────
# SETUP: Train XGBoost model
# ─────────────────────────────────────────────
print("=" * 65)
print("SETUP: Preparing Tools")
print("=" * 65)

print("Training XGBoost model...")
data = load_breast_cancer()
X    = data.data
y    = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)
xgb_model.fit(X_train, y_train)
feature_names = list(data.feature_names)
print("  XGBoost model ready.")


# ─────────────────────────────────────────────
# SETUP: Load RAG vectorstore
# ─────────────────────────────────────────────
print("Loading RAG vectorstore from disk...")
embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print(f"  Vectorstore ready. ({vectorstore._collection.count()} vectors)")


# ─────────────────────────────────────────────
# SETUP: Build the RAG chain
# ─────────────────────────────────────────────
rag_llm = ChatOllama(model=LLM_MODEL, temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a clinical decision support assistant for Riverside Medical Center.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have that information in the current guidelines."
Be concise and specific.

Context:
{context}

Question: {question}

Answer:"""
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | rag_llm
    | StrOutputParser()
)


# ─────────────────────────────────────────────
# TOOL 1: RAG — search clinical guidelines
# ─────────────────────────────────────────────
# Using Tool class instead of @tool decorator so the LLM passes a plain
# string with no named field — eliminates parameter name mismatch errors.

def _search_guidelines(query) -> str:
    # LLMs sometimes pass a dict instead of a plain string
    if isinstance(query, dict):
        query = str(list(query.values())[0])
    return rag_chain.invoke(str(query))

search_guidelines = Tool(
    name="search_guidelines",
    description=(
        "Search Riverside Medical Center clinical guidelines to answer questions "
        "about treatments, medications, dosages, drug interactions, and clinical "
        "protocols. Input is a plain string question. Use this for any policy or "
        "guideline question."
    ),
    func=_search_guidelines,
)


# ─────────────────────────────────────────────
# TOOL 2: XGBoost — predict tumor malignancy
# ─────────────────────────────────────────────

def _predict_tumor(measurements) -> str:
    # Handle dict passed directly from LLM — use it as-is
    if isinstance(measurements, dict):
        parsed = measurements
    else:
        parsed = None

        # Attempt 1: parse as-is (also handles double-encoded JSON)
        try:
            parsed = json.loads(measurements)
            if isinstance(parsed, str):  # double-encoded — parse the inner string
                parsed = json.loads(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

        # Attempt 2: strip outer quotes and unescape backslashes
        if parsed is None:
            try:
                cleaned = measurements.strip().strip('"').replace('\\"', '"')
                parsed = json.loads(cleaned)
            except (json.JSONDecodeError, TypeError):
                pass

    if parsed is None:
        return "Error: could not parse measurements. Please pass a valid JSON string."

    # Build full 30-feature vector, filling unknowns with training mean
    baseline = np.mean(X_train, axis=0)
    features = baseline.copy()

    for key, value in parsed.items():
        if key in feature_names:
            idx = feature_names.index(key)
            features[idx] = float(value)

    features_2d = features.reshape(1, -1)
    prediction  = xgb_model.predict(features_2d)[0]
    probability = xgb_model.predict_proba(features_2d)[0]
    label       = "Benign" if prediction == 1 else "Malignant"
    confidence  = max(probability) * 100

    return (
        f"Prediction: {label}\n"
        f"Confidence: {confidence:.1f}%\n"
        f"P(Malignant): {probability[0]:.4f}\n"
        f"P(Benign):    {probability[1]:.4f}"
    )

predict_tumor = Tool(
    name="predict_tumor",
    description=(
        "Predict whether a breast tumor is malignant or benign using patient "
        "measurements. Input must be a JSON string with these keys: "
        "radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, "
        "compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, "
        "fractal_dimension_mean. Returns prediction and confidence score."
    ),
    func=_predict_tumor,
)


# ─────────────────────────────────────────────
# CREATE THE AGENT
# ─────────────────────────────────────────────
print("Initializing agent...")

# System prompt tells the agent to always use tools and never answer
# from its own training knowledge. Also explicitly instructs it to
# write its final answer in plain English — not as raw JSON.
system_prompt = SystemMessage(content="""You are a healthcare AI assistant.
You MUST use the provided tools to answer every question.
Never answer from your own training knowledge — always call the appropriate tool.
If a question has multiple parts that require different tools,
call EACH tool separately before giving a final answer.
After all tools have returned results, write your final answer in plain English.""")

agent_llm = ChatOllama(model=LLM_MODEL, temperature=0)
tools     = [search_guidelines, predict_tumor]
agent     = create_react_agent(agent_llm, tools, prompt=system_prompt)

print("  Agent ready.\n")


# ─────────────────────────────────────────────
# HELPER: Run a query and print the agent's steps
# ─────────────────────────────────────────────
def run_agent(query: str):
    print("─" * 65)
    print(f"QUERY: {query}")
    print("─" * 65)

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    for msg in result["messages"]:
        msg_type = type(msg).__name__

        if msg_type == "HumanMessage":
            pass

        elif msg_type == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n  → Agent calls tool : {tc['name']}")
                    args_preview = json.dumps(tc["args"])[:200]
                    print(f"    Input            : {args_preview}")
            elif msg.content:
                # Skip lines where the LLM accidentally outputs raw JSON
                # tool-call syntax instead of a natural language answer
                content = msg.content.strip()
                if not content.startswith('{"name"'):
                    print(f"\nFINAL ANSWER:\n{content}")

        elif msg_type == "ToolMessage":
            preview = msg.content[:300].replace("\n", " ")
            print(f"    Tool returned    : {preview}...")

    print()


# ─────────────────────────────────────────────
# RUN THE AGENT ON 3 QUERIES
# ─────────────────────────────────────────────
print("=" * 65)
print("DEMO 1: Query that needs the RAG tool only")
print("=" * 65)
run_agent(
    "What is the HbA1c target for a diabetic patient over 75 years old "
    "at Riverside Medical Center?"
)

print("=" * 65)
print("DEMO 2: Query that needs the XGBoost tool only")
print("=" * 65)
run_agent(
    'A patient has these tumor measurements: '
    '{"radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8, '
    '"area_mean": 1001.0, "smoothness_mean": 0.1184, "compactness_mean": 0.2776, '
    '"concavity_mean": 0.3001, "concave_points_mean": 0.1471, '
    '"symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871}. '
    "Is this tumor likely malignant or benign?"
)

print("=" * 65)
print("DEMO 3: Query that needs both tools")
print("=" * 65)
run_agent(
    'A patient has a suspicious tumor with measurements: '
    '{"radius_mean": 13.54, "texture_mean": 14.36, "perimeter_mean": 87.46, '
    '"area_mean": 566.3, "smoothness_mean": 0.09779, "compactness_mean": 0.08129, '
    '"concavity_mean": 0.06664, "concave_points_mean": 0.04781, '
    '"symmetry_mean": 0.1885, "fractal_dimension_mean": 0.05766}. '
    "Predict whether it is malignant or benign, and also tell me what the "
    "diabetes HbA1c target is for elderly patients at Riverside Medical Center."
)

print("=" * 65)
print("Done. Agent demo complete.")
print("=" * 65)
