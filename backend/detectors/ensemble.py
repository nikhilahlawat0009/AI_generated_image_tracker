"""
Ensemble scorer: combines model, frequency, and metadata detector scores
into a single confidence score with a human-readable verdict.

Weights are calibrated empirically:
  - Model score carries most signal for photorealistic AI images
  - Metadata is highly reliable when present (hard evidence)
  - Frequency provides supplementary signal especially for GAN images
"""
from dataclasses import dataclass


WEIGHTS = {
    "model": 0.50,
    "frequency": 0.25,
    "metadata": 0.25,
}

# If metadata gives a definitive signal (generator found), override
METADATA_HARD_EVIDENCE_THRESHOLD = 0.95


@dataclass
class EnsembleResult:
    is_ai_generated: str        # "yes" | "no" | "uncertain"
    confidence: float           # 0.0 – 1.0
    model_score: float
    frequency_score: float
    metadata_score: float
    evidence_flags: list[str]
    verdict_explanation: str


def score(
    model_result: dict,
    frequency_result: dict,
    metadata_result: dict,
) -> EnsembleResult:
    m = model_result.get("score", 0.5)
    f = frequency_result.get("score", 0.5)
    md = metadata_result.get("score", 0.5)

    all_flags = (
        frequency_result.get("flags", []) +
        metadata_result.get("flags", [])
    )

    # Hard evidence override: if metadata conclusively identifies generator
    if md >= METADATA_HARD_EVIDENCE_THRESHOLD:
        composite = 0.97
        explanation = (
            "Metadata contains definitive AI generator signature. "
            "This is hard evidence — the file's embedded data explicitly identifies an AI tool."
        )
    else:
        composite = (
            WEIGHTS["model"] * m +
            WEIGHTS["frequency"] * f +
            WEIGHTS["metadata"] * md
        )
        explanation = _build_explanation(m, f, md, composite, all_flags)

    composite = round(min(1.0, max(0.0, composite)), 4)

    if composite >= 0.70:
        verdict = "yes"
    elif composite <= 0.35:
        verdict = "no"
    else:
        verdict = "uncertain"

    return EnsembleResult(
        is_ai_generated=verdict,
        confidence=composite,
        model_score=round(m, 4),
        frequency_score=round(f, 4),
        metadata_score=round(md, 4),
        evidence_flags=all_flags,
        verdict_explanation=explanation,
    )


def _build_explanation(m: float, f: float, md: float, composite: float, flags: list) -> str:
    parts = []

    if m >= 0.75:
        parts.append(f"the ML classifier is highly confident this is AI-generated ({m:.0%})")
    elif m >= 0.55:
        parts.append(f"the ML classifier leans toward AI ({m:.0%})")
    elif m <= 0.35:
        parts.append(f"the ML classifier suggests this is likely a real photo ({m:.0%} AI probability)")

    if f >= 0.6:
        parts.append("frequency analysis found spectral artifacts consistent with AI generation")
    elif f <= 0.2:
        parts.append("frequency spectrum looks natural")

    if md >= 0.6:
        parts.append("metadata analysis found anomalies (missing or suspicious EXIF)")
    elif md <= 0.2:
        parts.append("metadata looks consistent with a real camera")

    if not parts:
        return "Signals are mixed — the image shows some AI characteristics but also natural properties."

    return "Analysis shows: " + "; ".join(parts) + "."
