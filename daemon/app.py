"""FastAPI Backend Daemon for Spark-Foresight."""

from __future__ import annotations

import asyncio
import logging
import uuid
import torch
from contextlib import asynccontextmanager
from typing import Optional

import platform
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_INDEX_PATH = STATIC_DIR / "index.html"

from daemon.audio.capture import discover_audio_nodes
from daemon.asr.engine import ASREngine
from daemon.db.repository import DatabaseRepository
from daemon.session_manager import LiveSession
from daemon.llm.client import GemmaClient
from daemon.rag.embeddings import EmbeddingGemmaClient
from daemon.sync.enricher import ShorthandEnricher
from daemon.rag.vector_store import VectorStore
from daemon.rag.watcher import FolderWatcherService
from daemon.rag.assistant import LiveAssistant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("spark_foresight")

repository = DatabaseRepository()
asr_engine: Optional[ASREngine] = None
active_sessions: dict[str, LiveSession] = {}
gemma_client = GemmaClient(model="gemma4:e4b")
embed_client = EmbeddingGemmaClient(model="embeddinggemma-2:latest")
enricher = ShorthandEnricher(repository=repository, llm_client=gemma_client)
vector_store = VectorStore(embedding_client=embed_client)
assistant = LiveAssistant(
    vector_store=vector_store, repository=repository, llm_client=gemma_client
)
folder_watcher = FolderWatcherService(vector_store=vector_store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_engine
    logger.info("Initializing Spark-Foresight Daemon...")
    await repository.initialize()

    # Load ASR engine once at startup on GB10 CUDA
    asr_engine = ASREngine(model_id="openai/whisper-tiny")

    # Start folder watcher on default docs directory
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    folder_watcher.start_watching(docs_dir)

    logger.info("Spark-Foresight Daemon initialized and ready.")
    yield
    # Cleanup on shutdown
    folder_watcher.stop()
    for session_id, live_session in list(active_sessions.items()):
        await live_session.stop()
    active_sessions.clear()


app = FastAPI(
    title="OpenForesight Daemon",
    description="100% On-Device Meeting & Knowledge Intelligence for Linux (NVIDIA / PipeWire / Gemma 4 / Ollama)",
    version="0.1.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartSessionRequest(BaseModel):
    title: str = "Untitled Meeting"
    mic_node: Optional[str] = None
    sink_node: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def serve_studio():
    """Serve the interactive Spark-Foresight Web Studio UI."""
    if STATIC_INDEX_PATH.exists():
        with open(STATIC_INDEX_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>OpenForesight Daemon Running. Visit <a href='/docs'>/docs</a>.</h2>")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/api/health")
async def health_check():
    """System health check and GPU telemetry."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU Only"
    arch = platform.machine()
    return {
        "status": "healthy",
        "platform": f"Linux ({arch})",
        "cuda_available": cuda_available,
        "device": device_name,
        "active_sessions_count": len(active_sessions),
    }


@app.get("/api/audio/devices")
async def get_audio_devices():
    """Enumerate PipeWire audio sources and sinks."""
    nodes = await asyncio.to_thread(discover_audio_nodes)
    return {
        "devices": [
            {
                "id": n.id,
                "name": n.name,
                "description": n.description,
                "class": n.media_class,
            }
            for n in nodes
        ]
    }


@app.post("/api/sessions/start")
async def start_session(req: StartSessionRequest):
    """Start a new recording and transcription session."""
    session_id = str(uuid.uuid4())
    assert asr_engine is not None, "ASR engine not initialized"

    live_session = LiveSession(
        session_id=session_id,
        title=req.title,
        repository=repository,
        asr_engine=asr_engine,
        mic_node=req.mic_node,
        sink_node=req.sink_node,
    )

    await live_session.start()
    active_sessions[session_id] = live_session

    return {
        "session_id": session_id,
        "title": req.title,
        "started_at": live_session.start_epoch_ms,
        "status": "recording",
        "mic_node": live_session.mic_node,
        "sink_node": live_session.sink_node,
    }


@app.post("/api/sessions/{session_id}/stop")
async def stop_session(session_id: str):
    """Stop an ongoing meeting session."""
    live_session = active_sessions.pop(session_id, None)
    if live_session:
        await live_session.stop()
    else:
        # Check DB if already recorded
        session = await repository.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        await repository.stop_session(session_id)

    return {"session_id": session_id, "status": "stopped"}


@app.get("/api/sessions")
async def list_sessions():
    """Retrieve list of all meeting sessions."""
    return {"sessions": await repository.list_sessions()}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve specific session info."""
    session = await repository.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session}


@app.get("/api/sessions/{session_id}/transcripts")
async def get_session_transcripts(
    session_id: str,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
):
    """Fetch transcript segments for a meeting."""
    segments = await repository.get_transcript_segments(
        session_id=session_id, start_ms=start_ms, end_ms=end_ms
    )
    return {"session_id": session_id, "count": len(segments), "segments": segments}


@app.get("/api/sessions/{session_id}/window")
async def get_temporal_window(
    session_id: str,
    center_ms: int,
    window_before_ms: int = 45000,
    window_after_ms: int = 15000,
):
    """
    Retrieve transcript segments within [center_ms - 45s, center_ms + 15s].
    Core API for the Shorthand Note Alignment Engine.
    """
    segments = await repository.get_temporal_window(
        session_id=session_id,
        center_ms=center_ms,
        window_before_ms=window_before_ms,
        window_after_ms=window_after_ms,
    )
    return {
        "session_id": session_id,
        "center_ms": center_ms,
        "window": f"[{center_ms - window_before_ms}ms, {center_ms + window_after_ms}ms]",
        "segments": segments,
    }


class SaveNotesRequest(BaseModel):
    raw_notes: str
    note_timestamps: Optional[dict[str, int]] = None


class EnrichNotesRequest(BaseModel):
    raw_notes: Optional[str] = None
    note_timestamps: Optional[dict[str, int]] = None
    model: Optional[str] = "gemma4:e4b"


@app.get("/api/models")
async def get_models_status():
    """Report status of Google Foresight models running on Ollama."""
    gemma_ready = await gemma_client.is_ready()
    return {
        "llm": gemma_client.model,
        "llm_ready": gemma_ready,
        "embedding": embed_client.model,
        "embedding_dimensions": 768,
        "provider": "Local Ollama on NVIDIA GPU",
    }


@app.post("/api/sessions/{session_id}/notes")
async def save_session_notes(session_id: str, req: SaveNotesRequest):
    """Save user shorthand notes."""
    session = await repository.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    note_id = f"note-{session_id}"
    await repository.save_note(
        session_id=session_id,
        note_id=note_id,
        raw_text=req.raw_notes,
        created_at_ms=0,
    )
    return {"status": "saved", "session_id": session_id}


@app.get("/api/sessions/{session_id}/notes")
async def get_session_notes(session_id: str):
    """Retrieve saved notes for a session."""
    notes = await repository.get_notes(session_id)
    return {"session_id": session_id, "notes": notes}


@app.post("/api/sessions/{session_id}/notes/enrich")
async def enrich_session_notes(session_id: str, req: EnrichNotesRequest):
    """
    The Core Foresight Shorthand Engine:
    Correlates user notes with conversation windows and synthesizes grounded minutes using Gemma 4.
    """
    session = await repository.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # If raw_notes not provided in request, load from DB
    raw_notes = req.raw_notes
    if not raw_notes:
        existing = await repository.get_notes(session_id)
        if existing:
            raw_notes = existing[-1]["raw_text"]

    if not raw_notes:
        raise HTTPException(status_code=400, detail="No notes provided to enrich.")

    # Pass model override per-request without mutating global client
    target_model = req.model or enricher.llm.model

    enriched_markdown = await enricher.enrich_meeting_notes(
        session_id=session_id,
        raw_notes=raw_notes,
        note_timestamps=req.note_timestamps,
        model=target_model,
    )

    # Persist enriched output to DB
    note_id = f"note-{session_id}"
    await repository.save_note(
        session_id=session_id,
        note_id=note_id,
        raw_text=raw_notes,
        created_at_ms=0,
        enriched_text=enriched_markdown,
        status="enriched",
    )

    return {
        "session_id": session_id,
        "model_used": target_model,
        "enriched_markdown": enriched_markdown,
    }


class IndexKnowledgeRequest(BaseModel):
    path: str


class SearchKnowledgeRequest(BaseModel):
    query: str
    top_k: int = 5


class AssistantChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


@app.post("/api/knowledge/index")
async def index_knowledge_path(req: IndexKnowledgeRequest):
    """Index a file or folder into the 768-D VectorStore."""
    target_path = Path(req.path).resolve()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Path does not exist")

    # Reject system root or root dir traversal
    if target_path in [Path("/"), Path("/etc"), Path("/var"), Path("/root")]:
        raise HTTPException(status_code=403, detail="Indexing restricted system directories is not permitted.")

    if target_path.is_file():
        res = await vector_store.index_file(target_path)
        return {"status": "indexed", "results": [res]}
    elif target_path.is_dir():
        results = await vector_store.index_directory(target_path)
        folder_watcher.start_watching(target_path)
        return {"status": "indexed", "results": results}
    else:
        raise HTTPException(status_code=400, detail="Invalid path type")


@app.get("/api/knowledge/documents")
async def get_knowledge_documents():
    """List indexed documents in the local vector store."""
    docs = await vector_store.list_documents()
    return {"documents": docs, "total_documents": len(docs)}


@app.delete("/api/knowledge/documents/{doc_id}")
async def delete_knowledge_document(doc_id: str):
    """Remove a document from the vector store."""
    success = await vector_store.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "document_id": doc_id}


@app.post("/api/knowledge/search")
async def search_knowledge(req: SearchKnowledgeRequest):
    """Execute 768-D cosine similarity search using EmbeddingGemma 2."""
    matches = await vector_store.similarity_search(query=req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "model": embed_client.model,
        "matches": [
            {
                "chunk_id": m.chunk_id,
                "file_name": m.file_name,
                "file_path": m.file_path,
                "section": m.section_title,
                "page": m.page_number,
                "score": round(m.score, 4),
                "text": m.text,
            }
            for m in matches
        ],
    }


@app.post("/api/assistant/chat")
async def assistant_chat(req: AssistantChatRequest):
    """
    Live Assistant Grounded Q&A:
    Retrieves Top-K document chunks & meeting transcripts, synthesizing answers with Gemma 4.
    """
    resp = await assistant.answer_query(query=req.query, session_id=req.session_id)
    return {
        "query": resp.query,
        "answer_markdown": resp.answer_markdown,
        "citations": [
            {
                "source_type": c.source_type,
                "source_label": c.source_label,
                "snippet": c.text_snippet,
                "score": round(c.score, 4),
            }
            for c in resp.citations
        ],
    }


@app.websocket("/ws/transcripts/{session_id}")
async def websocket_transcripts(websocket: WebSocket, session_id: str):
    """WebSocket stream emitting live transcript segments to UI."""
    await websocket.accept()
    live_session = active_sessions.get(session_id)

    if not live_session:
        # If session is historical, stream all existing segments and close
        existing = await repository.get_transcript_segments(session_id)
        for seg in existing:
            await websocket.send_json({"type": "transcript_segment", **seg})
        await websocket.close()
        return

    live_session.register_client(websocket)
    try:
        await websocket.send_json(
            {"type": "connected", "session_id": session_id, "status": "subscribed"}
        )
        while True:
            # Keep socket alive and listen for client heartbeats/messages
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        live_session.unregister_client(websocket)
    except Exception:
        live_session.unregister_client(websocket)
