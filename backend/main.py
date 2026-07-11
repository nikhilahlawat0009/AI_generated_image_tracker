import os
import uuid
import asyncio
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import aiofiles
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
import json

load_dotenv()

from db.database import init_db, get_db, ImageAnalysis
from detectors import model_detector, frequency_analyzer, metadata_analyzer
from detectors.ensemble import score as ensemble_score
from agents.provenance_agent import analyze as agent_analyze

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", 10)) * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

app = FastAPI(title="AI Image Tracker", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await init_db()
    logger.info("Database initialized.")


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    # Validate
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large (max {os.getenv('MAX_FILE_SIZE_MB', 10)}MB)")

    # Save upload
    analysis_id = str(uuid.uuid4())
    ext = Path(file.filename or "image").suffix or ".jpg"
    save_path = UPLOAD_DIR / f"{analysis_id}{ext}"
    async with aiofiles.open(save_path, "wb") as f:
        await f.write(image_bytes)

    # Load image
    try:
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    logger.info(f"Analyzing image {file.filename} ({len(image_bytes)} bytes)")

    # Run detectors in parallel (model is sync, wrap in executor)
    loop = asyncio.get_event_loop()

    model_result, freq_result, meta_result = await asyncio.gather(
        loop.run_in_executor(None, model_detector.detect, image),
        loop.run_in_executor(None, frequency_analyzer.detect, image),
        loop.run_in_executor(None, metadata_analyzer.detect, image, image_bytes),
    )

    # Ensemble
    result = ensemble_score(model_result, freq_result, meta_result)

    # Claude agent analysis
    agent_report = await agent_analyze(
        image, model_result, freq_result, meta_result, result.confidence
    )

    # Persist
    record = ImageAnalysis(
        id=analysis_id,
        filename=file.filename or "unknown",
        file_path=str(save_path),
        uploaded_at=datetime.utcnow(),
        is_ai_generated=result.is_ai_generated,
        confidence=result.confidence,
        model_score=result.model_score,
        frequency_score=result.frequency_score,
        metadata_score=result.metadata_score,
        detector_details={
            "model": model_result,
            "frequency": {"score": freq_result["score"], "details": freq_result.get("details", {})},
            "metadata": {"score": meta_result["score"], "details": meta_result.get("details", {})},
        },
        agent_report=agent_report,
        evidence_flags=result.evidence_flags,
    )
    db.add(record)
    await db.commit()

    return {
        "id": analysis_id,
        "filename": file.filename,
        "is_ai_generated": result.is_ai_generated,
        "confidence": result.confidence,
        "scores": {
            "model": result.model_score,
            "frequency": result.frequency_score,
            "metadata": result.metadata_score,
        },
        "verdict_explanation": result.verdict_explanation,
        "evidence_flags": result.evidence_flags,
        "agent_report": agent_report,
        "uploaded_at": record.uploaded_at.isoformat(),
    }


@app.get("/api/history")
async def get_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ImageAnalysis).order_by(desc(ImageAnalysis.uploaded_at)).limit(limit)
    result = await db.execute(stmt)
    records = result.scalars().all()
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "is_ai_generated": r.is_ai_generated,
            "confidence": r.confidence,
            "uploaded_at": r.uploaded_at.isoformat() if r.uploaded_at else None,
        }
        for r in records
    ]


@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.get(ImageAnalysis, analysis_id)
    if not result:
        raise HTTPException(404, "Analysis not found")
    return {
        "id": result.id,
        "filename": result.filename,
        "is_ai_generated": result.is_ai_generated,
        "confidence": result.confidence,
        "scores": {
            "model": result.model_score,
            "frequency": result.frequency_score,
            "metadata": result.metadata_score,
        },
        "evidence_flags": result.evidence_flags,
        "agent_report": result.agent_report,
        "detector_details": result.detector_details,
        "uploaded_at": result.uploaded_at.isoformat() if result.uploaded_at else None,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}
