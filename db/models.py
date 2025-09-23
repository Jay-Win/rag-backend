# db/models.py
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, ForeignKey, JSON, Text, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Chat(Base):
    __tablename__ = "chats"
    id = Column(String(36), primary_key=True)          # UUID string
    title = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    archived = Column(Boolean, default=False, nullable=False)

    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_chats_created_at", "created_at"),
        Index("ix_chats_archived", "archived"),
    )

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(16), nullable=False)          # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)             # what you show
    raw = Column(Text)                                 # full stdout blob
    parsed_response = Column(Text)                     # sanitized HTML/string you render
    sources = Column(JSON)                             # list[str]
    payload = Column(JSON)                             # request payload sent
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    chat = relationship("Chat", back_populates="messages")

    __table_args__ = (Index("ix_messages_chat_created", "chat_id", "created_at"),)
