"""
ML-based AI image detector using a fine-tuned ViT classifier from HuggingFace.
Model: umm-maybe/AI-image-detector (CLIP ViT-L/14 fine-tuned on real vs AI images)
Returns a probability score 0.0 (real) → 1.0 (AI generated).
"""
from transformers import pipeline
from PIL import Image
import torch
import logging

logger = logging.getLogger(__name__)

_pipe = None


def _get_pipeline():
    global _pipe
    if _pipe is None:
        logger.info("Loading AI image detector model...")
        device = 0 if torch.cuda.is_available() else -1
        _pipe = pipeline(
            "image-classification",
            model="umm-maybe/AI-image-detector",
            device=device,
        )
        logger.info("Model loaded.")
    return _pipe


def detect(image: Image.Image) -> dict:
    """
    Returns:
        score: float 0.0–1.0, probability image is AI-generated
        label: "AI" | "Real"
        raw: full pipeline output
    """
    try:
        pipe = _get_pipeline()
        results = pipe(image.convert("RGB"))
        # pipeline returns list of {label, score}
        ai_score = next(
            (r["score"] for r in results if r["label"].lower() in ("artificial", "ai")),
            None,
        )
        if ai_score is None:
            # fallback: take the top label
            top = results[0]
            ai_score = top["score"] if top["label"].lower() not in ("real", "human") else 1 - top["score"]

        return {
            "score": round(float(ai_score), 4),
            "label": "AI" if ai_score >= 0.5 else "Real",
            "raw": results,
        }
    except Exception as e:
        logger.error(f"Model detector error: {e}")
        return {"score": 0.5, "label": "uncertain", "raw": [], "error": str(e)}
