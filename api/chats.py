# api/chats.py
import uuid
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional, List
from db.session import SessionLocal
from db.models import Chat, Message
from datetime import datetime
from api.security import check_key

router = APIRouter(prefix="/chats", tags=["chats"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Schemas
class ChatOut(BaseModel):
    id: str
    title: str
    archived: bool
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True

class ChatCreate(BaseModel):
    title: Optional[str] = None

class ChatRename(BaseModel):
    title: str

class MessageIn(BaseModel):
    role: str             # "user" | "assistant"
    content: str
    raw: Optional[str] = None
    parsed_response: Optional[str] = None
    sources: Optional[List[str]] = None
    payload: Optional[dict] = None

class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    raw: Optional[str]
    parsed_response: Optional[str]
    sources: Optional[List[str]]
    payload: Optional[dict]
    created_at: datetime
    class Config:
        from_attributes = True



@router.get("", response_model=List[ChatOut])
def list_chats(
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
    q: Optional[str] = Query(None),
    archived: Optional[bool] = Query(False),
    limit: int = 100,
):
    check_key(x_api_key)
    qry = db.query(Chat).filter(Chat.archived == (archived or False))
    if q:
        qry = qry.filter(Chat.title.ilike(f"%{q}%"))
    return qry.order_by(Chat.updated_at.desc()).limit(limit).all()

@router.post("", response_model=ChatOut)
def create_chat(
    body: ChatCreate,
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    check_key(x_api_key)
    cid = str(uuid.uuid4())
    title = body.title or "New Chat"
    chat = Chat(id=cid, title=title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat

@router.get("/{chat_id}", response_model=ChatOut)
def get_chat(chat_id: str, x_api_key: Optional[str] = Header(None), db: Session = Depends(get_db)):
    check_key(x_api_key)
    chat = db.query(Chat).get(chat_id)
    if not chat: raise HTTPException(404, "Chat not found")
    return chat

@router.patch("/{chat_id}", response_model=ChatOut)
def rename_chat(chat_id: str, body: ChatRename, x_api_key: Optional[str] = Header(None), db: Session = Depends(get_db)):
    check_key(x_api_key)
    chat = db.query(Chat).get(chat_id)
    if not chat: raise HTTPException(404, "Chat not found")
    chat.title = body.title
    chat.updated_at = datetime.utcnow()
    db.commit(); db.refresh(chat)
    return chat

@router.delete("/{chat_id}")
def delete_chat(chat_id: str, x_api_key: Optional[str] = Header(None), db: Session = Depends(get_db)):
    check_key(x_api_key)
    chat = db.query(Chat).get(chat_id)
    if not chat: raise HTTPException(404, "Chat not found")
    db.delete(chat); db.commit()
    return {"ok": True}

@router.post("/{chat_id}/archive")
def archive_chat(chat_id: str, x_api_key: Optional[str] = Header(None), db: Session = Depends(get_db)):
    check_key(x_api_key)
    chat = db.query(Chat).get(chat_id)
    if not chat: raise HTTPException(404, "Chat not found")
    chat.archived = True; chat.updated_at = datetime.utcnow()
    db.commit()
    return {"ok": True}

@router.get("/{chat_id}/messages", response_model=List[MessageOut])
def list_messages(chat_id: str, x_api_key: Optional[str] = Header(None), db: Session = Depends(get_db)):
    check_key(x_api_key)
    chat = db.query(Chat).get(chat_id)
    if not chat: raise HTTPException(404, "Chat not found")
    msgs = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at.asc()).all()
    return msgs

@router.post("/{chat_id}/messages", response_model=MessageOut)
def add_message(chat_id: str, body: MessageIn, x_api_key: Optional[str] = Header(None), db: Session = Depends(get_db)):
    check_key(x_api_key)
    chat = db.query(Chat).get(chat_id)
    if not chat: raise HTTPException(404, "Chat not found")
    msg = Message(
        chat_id=chat_id, role=body.role, content=body.content,
        raw=body.raw, parsed_response=body.parsed_response,
        sources=body.sources, payload=body.payload
    )
    chat.updated_at = datetime.utcnow()
    db.add(msg); db.commit(); db.refresh(msg)
    return msg
