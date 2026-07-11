"""
Claude-powered provenance agent.

Uses Claude's vision + tool use to:
1. Visually inspect the image for AI tell-tale signs
2. Synthesize detector evidence into a human-readable report
3. Identify the likely AI generator/model if applicable
"""
import base64
import io
import os
import logging
from PIL import Image
import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert forensic analyst specializing in detecting AI-generated images.
You will be given an image and detector analysis results. Your job is to:

1. Visually inspect the image for AI generation artifacts:
   - Unnatural textures (skin, hair, fabric)
   - Impossible lighting or shadow inconsistencies
   - Anatomical errors (extra fingers, distorted hands/ears/eyes)
   - Background incoherence or "dreamlike" blur
   - Overly perfect or uncanny symmetry
   - Watermarks or style signatures from known AI tools

2. Synthesize the detector evidence provided to you

3. Output a structured forensic report with:
   - Your visual assessment
   - Specific artifacts you observed (with location if possible)
   - Likely AI generator/model if identifiable
   - Your confidence level and reasoning
   - A one-sentence verdict

Be precise and evidence-based. Do not speculate beyond what you observe."""


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


async def analyze(
    image: Image.Image,
    model_result: dict,
    frequency_result: dict,
    metadata_result: dict,
    ensemble_confidence: float,
) -> str:
    """
    Returns a markdown-formatted forensic report string.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "_Agent analysis unavailable: ANTHROPIC_API_KEY not set._"

    client = anthropic.Anthropic(api_key=api_key)

    detector_summary = f"""
## Automated Detector Results

**ML Classifier Score**: {model_result.get('score', 'N/A')} (label: {model_result.get('label', 'N/A')})

**Frequency Analysis Score**: {frequency_result.get('score', 'N/A')}
Flags: {'; '.join(frequency_result.get('flags', [])) or 'None'}

**Metadata Analysis Score**: {metadata_result.get('score', 'N/A')}
Flags: {'; '.join(metadata_result.get('flags', [])) or 'None'}

**Ensemble Confidence**: {ensemble_confidence:.1%}
"""

    user_message = f"""Please analyze this image for AI generation.

{detector_summary}

Provide your forensic analysis covering:
1. Visual artifacts you observe directly in the image
2. Your interpretation of the detector evidence above
3. Likely AI tool/model if identifiable
4. Overall confidence and verdict

Format as a clear forensic report."""

    try:
        image_b64 = _image_to_base64(image)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": user_message},
                    ],
                }
            ],
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Agent analysis error: {e}")
        return f"_Agent analysis failed: {e}_"
