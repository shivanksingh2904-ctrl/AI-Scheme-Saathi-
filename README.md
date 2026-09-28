# SchemeSaathi

**Describe your situation in plain language and find government schemes and scholarships you may qualify for, with answers grounded in official sources.**

> **Status:** starter dataset of 9 schemes. Data is not yet verified against official pages; see "Limitations".

## Problem

Information about government schemes, scholarships and loans is scattered across many portals and PDFs. Eligibility rules are hard to read, so many people never find schemes they may qualify for.

## Solution

SchemeSaathi lets a user type their situation in their own words (for example, "I'm a 20-year-old girl studying engineering and my family income is under 3 lakh"). It retrieves the most relevant scheme information from a curated knowledge base, then uses an LLM to explain, in simple language, which schemes may apply, which documents are needed and how to apply, with source links.

The official [myScheme](https://www.myscheme.gov.in) portal already helps users discover schemes through a form. SchemeSaathi adds a **free-text, conversational layer with plain-language explanations and cited sources**. It is not a replacement for the official portal.

## Features

- Natural-language questions about schemes, scholarships, loans and benefits
- Semantic search using embeddings (finds matches even when wording differs)
- Answers grounded only in retrieved context, with source links
- Says "not found" instead of guessing when nothing relevant exists
- Works without an LLM key (retrieval-only mode)
- Input validation, rate limiting, and prompt-injection-aware prompt
- Simple web UI served by the same backend

## Tech stack

| Tech | Purpose |
|---|---|
| Python | Application language |
| FastAPI + Pydantic | REST API and request/response validation |
| all-MiniLM-L6-v2 on ONNX Runtime (Chroma's default embedding function) | Converts text to embeddings locally, without PyTorch |
| ChromaDB | Vector store with metadata filtering |
| Any OpenAI-compatible LLM API | Generates the final grounded answer |
| requests | HTTP calls to the LLM API |
| python-dotenv | Loads secrets from `.env` |
| HTML/CSS/JS | Minimal frontend |
| pytest | Basic tests |

No LangChain or LlamaIndex: the RAG pipeline is written by hand (about 150 lines) to keep every step visible.

## Architecture

```
User -> Frontend (index.html) -> POST /ask -> FastAPI (validation, rate limit)
     -> Retriever: embed question -> ChromaDB similarity search -> top chunks
     -> Prompt builder: rules + retrieved context + question
     -> LLM API -> answer
     -> JSON {answer, sources} -> Frontend -> User
```

Offline step (run once, and after any data change):
`data/schemes.json -> chunk -> embed -> store in ChromaDB`

## How RAG works here

1. **Chunking:** each scheme is split into 4 chunks (overview, eligibility, benefits, how to apply). Each chunk repeats the scheme name so it makes sense on its own.
2. **Embedding:** every chunk becomes a 384-number vector that captures its meaning.
3. **Retrieval:** the question is embedded the same way, and ChromaDB returns the chunks with the closest vectors (cosine distance). Chunks beyond a distance threshold are dropped.
4. **Augmented generation:** the retrieved chunks are placed in the prompt as context, and the LLM is told to answer only from them.
5. **Citation:** the source URL of each used scheme is returned to the user.

## How the LLM is used

The LLM is called through a standard chat-completions HTTP request with a system prompt (rules) and a user message (context + question) at low temperature (0.2) for factual answers. It never decides which documents exist; retrieval does. The LLM only rewrites retrieved facts in simple language.

## Installation

```bash
git clone https://github.com/<your-username>/schemesaathi.git
cd schemesaathi
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Environment setup

```bash
cp .env.example .env     # Windows: copy .env.example .env
```

Edit `.env` and add `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` from any OpenAI-compatible provider. Skip this to run in retrieval-only mode. Never commit `.env`.

## Running locally

```bash
python -m app.services.ingest      # build the vector database (first time and after data changes)
uvicorn main:app --reload          # start the app
```

Open http://127.0.0.1:8000. API docs are at http://127.0.0.1:8000/docs.

Run tests: `python -m pytest`

## Deploying on Render (free tier)

- Language: Python, Build Command: `pip install -r requirements.txt`
- Start Command: `python -m app.services.ingest && uvicorn main:app --host 0.0.0.0 --port $PORT`
- The repo includes `.python-version` to pin Python 3.11.
- Optional environment variables: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.
- Free instances sleep when idle, so the first request after a pause can take about a minute.

## Example queries

- "I am a 20-year-old girl studying engineering and my family income is under 3 lakh."
- "How can I get an education loan for my MBA without collateral?"
- "I am a small farmer with 1 acre of land. What support can I get?"
- "I want to open a small shop. Is there a loan without security?"

## Screenshots / demo

_Add screenshots here after running the app._

## Project structure

```
schemesaathi/
├── app/
│   ├── config.py          settings and env loading
│   ├── schemas.py         Pydantic request/response models
│   └── services/
│       ├── embedder.py    text -> vectors
│       ├── ingest.py      load data, chunk, embed, store
│       ├── retriever.py   question -> relevant chunks
│       ├── prompt.py      builds LLM messages
│       └── llm.py         LLM API call and error handling
├── data/schemes.json      curated knowledge base
├── static/index.html      frontend
├── tests/test_basic.py
├── main.py                FastAPI app and routes
├── requirements.txt
├── .env.example
└── .gitignore
```

## Limitations

- Small hand-curated dataset (9 schemes); no state-level schemes yet.
- Scheme details were written as a starter and must be verified against each official page before real use (`"verified": false` in the data).
- Scheme rules change; there is no automatic update.
- English only. Eligibility is explained, not decided.
- Not official, legal or financial advice.

## Future improvements

- Verify and expand data (state schemes, more scholarships) with a `last_checked` date per scheme
- Hindi and other Indian languages using a multilingual embedding model
- Small evaluation set of questions with expected schemes to measure retrieval accuracy
- Structured profile form (age, income, category) combined with free text
- Deployment on Hugging Face Spaces or Render

## Learning outcomes

Building a RAG pipeline without a framework, embeddings and cosine similarity, chunking trade-offs, prompt grounding against hallucination, REST API design with validation, secrets management, and basic security practices.

## Disclaimer

Information may be incomplete or outdated. Always confirm details on the official website before applying.
