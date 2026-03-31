# -*- coding: utf-8 -*-

import os
import uuid
import json
import time
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_pipeline import RAGPipeline

app = FastAPI(title="SmartDoc AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict = {}
session_meta: dict = {}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Models
class QueryRequest(BaseModel):
    session_id: str
    question: str
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str
    tokens_used: Optional[int] = None


class SessionInfo(BaseModel):
    session_id: str
    documents: List[str]
    created_at: float
    total_chunks: int


# Routes

@app.get("/")
def root():
    return {"status": "SmartDoc AI running", "version": "1.0.0"}


# ✅ FIXED FILE UPLOAD HERE
@app.post("/upload", response_model=SessionInfo)
async def upload_documents(files: List[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())
    saved_paths, doc_names = [], []

    for file in files:
        if not file.filename.endswith((".pdf", ".txt", ".md")):
            raise HTTPException(400, f"Unsupported file type: {file.filename}")

        dest = UPLOAD_DIR / f"{session_id}_{file.filename}"

        with open(dest, "wb") as f:
            content = await file.read()
            f.write(content)

        saved_paths.append(str(dest))
        doc_names.append(file.filename)

    pipeline = RAGPipeline()
    total_chunks = pipeline.ingest_documents(saved_paths)

    sessions[session_id] = pipeline
    session_meta[session_id] = {
        "documents": doc_names,
        "created_at": time.time(),
        "total_chunks": total_chunks,
    }

    return SessionInfo(
        session_id=session_id,
        documents=doc_names,
        created_at=session_meta[session_id]["created_at"],
        total_chunks=total_chunks,
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found. Upload documents first.")

    result = sessions[req.session_id].query(req.question)

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=req.session_id,
        tokens_used=result.get("tokens_used"),
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found.")

    pipeline = sessions[req.session_id]

    def token_generator():
        for token in pipeline.stream_query(req.question):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.get("/session/{session_id}", response_model=SessionInfo)
def get_session(session_id: str):
    if session_id not in session_meta:
        raise HTTPException(404, "Session not found.")

    return SessionInfo(session_id=session_id, **session_meta[session_id])


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    sessions.pop(session_id, None)
    session_meta.pop(session_id, None)
    return {"deleted": session_id}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "active_sessions": len(sessions),
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
    }