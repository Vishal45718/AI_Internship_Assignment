import json
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import shutil
from pathlib import Path
import os

from backend.app.storage.db import get_db, Conversation, Message
from src.pipeline import RAGPipeline
from src.config import get_settings

router = APIRouter()

# Global pipeline instance (lazy loading handled in pipeline itself)
_pipeline: Optional[RAGPipeline] = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

class ChatRequest(BaseModel):
    message: str
    mode: str = "document" # "general" or "document"
    conversation_id: str

@router.post("/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    conv_id = request.conversation_id
    if not conv_id:
        conv_id = str(uuid.uuid4())
        
    # Get or create conversation
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        conv = Conversation(id=conv_id, title=request.message[:50], mode=request.mode)
        db.add(conv)
        db.commit()

    # Save user message
    user_msg = Message(conversation_id=conv_id, role="user", content=request.message)
    db.add(user_msg)
    db.commit()

    # Retrieve history
    db_messages = db.query(Message).filter(Message.conversation_id == conv_id).order_by(Message.id).all()
    history = [{"role": msg.role, "content": msg.content} for msg in db_messages[:-1]] # Exclude the current message

    def event_stream():
        full_response = ""
        sources = []
        try:
            pipeline = get_pipeline()
            # We yield server-sent events (SSE)
            for chunk_type, data in pipeline.stream_query(request.message, mode=request.mode, history=history):
                if chunk_type == "token":
                    full_response += data
                    yield f"data: {json.dumps({'type': 'token', 'content': data})}\n\n"
                elif chunk_type == "sources":
                    sources = data
                    yield f"data: {json.dumps({'type': 'sources', 'content': data})}\n\n"
                elif chunk_type == "error":
                    full_response = data
                    yield f"data: {json.dumps({'type': 'error', 'content': data})}\n\n"
        except Exception as e:
            error_msg = str(e)
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            full_response = error_msg
        
        # Finally, save assistant message
        try:
            # We can't use the dependency 'db' in a generator reliably after the request finishes, 
            # so we create a new session locally for this final save.
            from backend.app.storage.db import SessionLocal
            with SessionLocal() as local_db:
                # Add sources to the end of the message if it's a document mode and we have sources
                # We save it as JSON in the database so the frontend can retrieve it later
                saved_content = full_response
                if sources:
                    saved_content = json.dumps({"text": full_response, "sources": sources})
                
                asst_msg = Message(conversation_id=conv_id, role="assistant", content=saved_content)
                local_db.add(asst_msg)
                
                # Update conversation updated_at
                local_conv = local_db.query(Conversation).filter(Conversation.id == conv_id).first()
                if local_conv:
                    local_conv.title = local_conv.title # Trigger onupdate
                
                local_db.commit()
        except Exception as e:
            print(f"Error saving message: {e}")
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    settings = get_settings()
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = data_dir / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    pipeline = get_pipeline()
    result = pipeline.ingest(file_path)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
        
    return {"status": "success", "file": file.filename, "chunks": result.get("chunks", 0)}

@router.get("/history")
async def get_history(db: Session = Depends(get_db)):
    convs = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    result = []
    for c in convs:
        result.append({
            "id": c.id,
            "title": c.title,
            "mode": c.mode,
            "updated_at": c.updated_at.isoformat()
        })
    return result

@router.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    messages = db.query(Message).filter(Message.conversation_id == conv_id).order_by(Message.id).all()
    return {
        "id": conv.id,
        "title": conv.title,
        "mode": conv.mode,
        "messages": [{"role": m.role, "content": m.content} for m in messages]
    }

@router.delete("/files/{filename}")
async def delete_file(filename: str):
    pipeline = get_pipeline()
    deleted = pipeline.delete_document(filename)
    
    settings = get_settings()
    file_path = settings.data_dir / filename
    if file_path.exists():
        os.remove(file_path)
        
    return {"status": "success", "deleted_chunks": deleted}

@router.get("/status")
async def get_status():
    pipeline = get_pipeline()
    stats = pipeline.get_stats()
    return stats
