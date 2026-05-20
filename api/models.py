"""SQLAlchemy ORM model for the Task table."""

from datetime import datetime

from sqlalchemy import Column, DateTime, String, Text

from api.db import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True)
    status = Column(String, nullable=False, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    video_path = Column(String, nullable=False)
    result_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
