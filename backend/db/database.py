from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy import Column, String, Float, DateTime, Text, JSON
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./tracker.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class ImageAnalysis(Base):
    __tablename__ = "image_analyses"

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    # Overall verdict
    is_ai_generated = Column(String)          # "yes" | "no" | "uncertain"
    confidence = Column(Float)                 # 0.0 - 1.0

    # Per-detector scores
    model_score = Column(Float)               # ML classifier
    frequency_score = Column(Float)           # FFT artifact score
    metadata_score = Column(Float)            # EXIF/metadata anomaly score

    # Supporting evidence
    detector_details = Column(JSON)           # raw outputs from each detector
    agent_report = Column(Text)               # Claude's narrative analysis
    evidence_flags = Column(JSON)             # list of specific evidence strings


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
