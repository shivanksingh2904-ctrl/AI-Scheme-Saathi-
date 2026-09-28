"""SchemeSaathi backend: FastAPI app tying the RAG pipeline together."""
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.schemas import AskRequest, AskResponse, Source
from app.services import llm, prompt, retriever
from app.services.embedder import embed_texts


@asynccontextmanager
async def lifespan(app: FastAPI):
    embed_texts(["warm up"])  # download/load the model now, not on the first request
    yield


app = FastAPI(title="SchemeSaathi", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

# --- very simple in-memory rate limiter: 10 requests per minute per IP ---
RATE_LIMIT = 10
WINDOW_SECONDS = 60
_hits = defaultdict(deque)


def check_rate_limit(ip: str) -> None:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > WINDOW_SECONDS:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute.")
    q.append(now)


@app.get("/")
def home():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/health")
def health():
    # Never return the key itself, only whether it is set.
    return {"status": "ok", "llm_configured": config.llm_configured()}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    check_rate_limit(ip)

    try:
        chunks = retriever.retrieve(body.question, category=body.category)
    except retriever.KnowledgeBaseNotReady:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not built yet. Run: python -m app.services.ingest",
        )

    if not chunks:
        return AskResponse(
            answer=(
                "I couldn't find a matching scheme in my database. Try describing your "
                "situation differently, or search on myscheme.gov.in."
            ),
            sources=[],
            mode="no_match",
        )

    # de-duplicate sources by scheme name, keeping order
    seen, sources = set(), []
    for c in chunks:
        if c["scheme"] not in seen:
            seen.add(c["scheme"])
            sources.append(Source(scheme=c["scheme"], url=c["source_url"]))

    if not config.llm_configured():
        text = "\n\n".join(c["text"] for c in chunks[:3])
        return AskResponse(
            answer="(AI answer is off because no LLM key is set. Here are the most relevant details.)\n\n" + text,
            sources=sources,
            mode="retrieval_only",
        )

    try:
        answer = llm.generate(prompt.build_messages(body.question, chunks))
    except llm.LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return AskResponse(answer=answer, sources=sources, mode="rag")
