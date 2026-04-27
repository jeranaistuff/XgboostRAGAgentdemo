# rag_demo.py
# Demonstrates Retrieval-Augmented Generation (RAG) for healthcare
#
# Stack:
#   - Documents  : Riverside Medical Center clinical guidelines (docs/ folder)
#   - Embeddings : Ollama (nomic-embed-text) — converts text to vectors locally
#   - Vector DB  : ChromaDB — stores and searches those vectors
#   - LLM        : Ollama (llama3.2) — generates the final answer
#   - Framework  : LangChain — ties all of the above together
#
# Before running:
#   ollama pull llama3.2
#   ollama pull nomic-embed-text

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DOCS_DIR    = "docs"
DB_DIR      = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "llama3.2"


# ─────────────────────────────────────────────
# 1. LOAD DOCUMENTS
# ─────────────────────────────────────────────
print("=" * 65)
print("STEP 1: Load Clinical Guideline Documents")
print("=" * 65)

loader = DirectoryLoader(
    DOCS_DIR,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
raw_docs = loader.load()

print(f"Documents loaded: {len(raw_docs)}")
for doc in raw_docs:
    filename = os.path.basename(doc.metadata["source"])
    print(f"  - {filename}  ({len(doc.page_content)} chars)")


# ─────────────────────────────────────────────
# 2. CHUNK THE DOCUMENTS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Split Documents into Chunks")
print("=" * 65)

# Why chunk? LLMs have a context window limit, and embedding a 5-page
# document as one unit produces a vague, averaged-out vector.
# Small focused chunks embed more precisely and retrieve more accurately.
#
# chunk_size    = max characters per chunk
# chunk_overlap = characters shared between adjacent chunks
#                 (prevents cutting a sentence mid-thought)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(raw_docs)

print(f"Total chunks created : {len(chunks)}")
print(f"Avg chunk size       : {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
print(f"\nExample chunk (from {os.path.basename(chunks[5].metadata['source'])}):")
print("-" * 65)
print(chunks[5].page_content)
print("-" * 65)


# ─────────────────────────────────────────────
# 3. EMBED AND STORE IN VECTOR DATABASE
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Embed Chunks and Store in ChromaDB")
print("=" * 65)

print(f"Embedding model : {EMBED_MODEL} (running via Ollama)")
print("Embedding chunks... (this may take ~20-30 seconds on first run)")

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# Chroma stores the vectors on disk in DB_DIR.
# On second run it reloads from disk instead of re-embedding.
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR,
)

print(f"Vector store ready. {vectorstore._collection.count()} vectors stored in '{DB_DIR}/'")


# ─────────────────────────────────────────────
# 4. SHOW RAW RETRIEVAL (before the LLM is involved)
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Demonstrate Retrieval (Semantic Search)")
print("=" * 65)

# This step isolates the retrieval component so you can see exactly
# what chunks the system finds before the LLM generates anything.

test_query = "What is the metformin dose for a new diabetes patient?"
retriever  = vectorstore.as_retriever(search_kwargs={"k": 3})
retrieved  = retriever.invoke(test_query)

print(f"Query: \"{test_query}\"")
print(f"\nTop {len(retrieved)} retrieved chunks:")
for i, doc in enumerate(retrieved, 1):
    source = os.path.basename(doc.metadata["source"])
    print(f"\n  [{i}] Source: {source}")
    print(f"  {doc.page_content[:300].strip()}...")


# ─────────────────────────────────────────────
# 5. WITHOUT RAG — LLM answers from memory alone
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: WITHOUT RAG — LLM Answering From Memory Only")
print("=" * 65)

# This shows what the LLM says with no context provided.
# For general medical knowledge it might do okay, but for
# Riverside MC-specific protocols (thresholds, extensions,
# specific drug preferences) it will have no way of knowing.

llm = OllamaLLM(model=LLM_MODEL)

specific_question = (
    "At Riverside Medical Center, what is the preferred diuretic "
    "for hypertension treatment and why?"
)

print(f"Question: \"{specific_question}\"")
print("\nLLM response (no context):")
print("-" * 65)
no_rag_response = llm.invoke(specific_question)
print(no_rag_response)


# ─────────────────────────────────────────────
# 6. WITH RAG — LLM answers using retrieved context
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: WITH RAG — LLM Answering Using Retrieved Context")
print("=" * 65)

# Build the RAG chain using LCEL (LangChain Expression Language).
# The | operator pipes output from one step into the next:
#
#   retriever        → fetches relevant chunks for the question
#   format_docs      → joins chunks into a single context string
#   prompt           → fills the template with context + question
#   llm              → generates the answer
#   StrOutputParser  → extracts the plain text from the LLM response

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_template = PromptTemplate(
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
    | prompt_template
    | llm
    | StrOutputParser()
)

print(f"Question: \"{specific_question}\"")
print("\nLLM response (with RAG context):")
print("-" * 65)
rag_response = rag_chain.invoke(specific_question)
print(rag_response)


# ─────────────────────────────────────────────
# 7. RUN A FEW MORE HEALTHCARE QUERIES
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7: Additional RAG Queries")
print("=" * 65)

queries = [
    "What are the contraindications for metformin?",
    "A patient is on warfarin and needs an antibiotic. What should I watch out for?",
    "What LDL target should a patient with known heart disease aim for?",
]

for q in queries:
    print(f"\nQ: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}")

print("\n" + "=" * 65)
print("Done. RAG demo complete.")
print("=" * 65)
