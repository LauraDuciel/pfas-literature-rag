import uvicorn
from fastapi import FastAPI, HTTPException

from pfas_lit_rag.answering import answer_question
from pfas_lit_rag.config import get_settings
from pfas_lit_rag.ingestion import build_index
from pfas_lit_rag.ollama_client import OllamaClient
from pfas_lit_rag.retrieval import search_index
from pfas_lit_rag.schemas import AnswerRequest, AnswerResponse, SearchRequest, SearchResponse

app = FastAPI(title="PFAS Literature RAG", version="0.1.0")


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    ollama_status = "unavailable"
    try:
        OllamaClient(settings).health()
        ollama_status = "available"
    except Exception:
        ollama_status = "unavailable"
    return {"status": "ok", "ollama": ollama_status, "model": settings.ollama_model}


@app.post("/ingest")
def ingest() -> dict:
    try:
        result = build_index(get_settings())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        results = search_index(request.query, settings=get_settings(), top_k=request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SearchResponse(query=request.query, results=results)


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    try:
        return answer_question(request.question, settings=get_settings(), top_k=request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc


def run() -> None:
    uvicorn.run("pfas_lit_rag.api:app", host="127.0.0.1", port=8000, reload=False)
