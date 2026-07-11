"""
EXIF / metadata forensics for AI image detection.

Real camera photos have rich, consistent metadata.
AI-generated images typically:
  - Have no EXIF data at all, OR
  - Have metadata injected by the generator tool (Stable Diffusion, Midjourney, DALL-E)
  - Lack camera-specific fields (make, model, focal length, GPS, etc.)
  - May contain generator watermarks in PNG tEXt chunks
"""
import io
import struct
import logging
from PIL import Image
from PIL.ExifTags import TAGS
import exifread

logger = logging.getLogger(__name__)

# Known AI generator metadata signatures
AI_GENERATOR_SIGNATURES = [
    "stable diffusion", "stablediffusion", "sd-webui",
    "midjourney", "dall-e", "dalle", "firefly", "adobe firefly",
    "diffusion", "comfyui", "invokeai", "novelai", "automatic1111",
    "generative", "ai generated", "artifical intelligence",
]

CAMERA_EXIF_FIELDS = {
    "Make", "Model", "LensModel", "FocalLength",
    "ExposureTime", "FNumber", "ISOSpeedRatings",
    "GPSInfo", "DateTimeOriginal",
}


def _read_png_text_chunks(image_bytes: bytes) -> dict:
    """Extract tEXt and iTXt chunks from PNG — AI tools often embed prompts here."""
    text_data = {}
    try:
        i = 8  # skip PNG signature
        while i < len(image_bytes):
            if i + 8 > len(image_bytes):
                break
            length = struct.unpack(">I", image_bytes[i: i + 4])[0]
            chunk_type = image_bytes[i + 4: i + 8].decode("ascii", errors="ignore")
            data = image_bytes[i + 8: i + 8 + length]
            if chunk_type in ("tEXt", "zTXt", "iTXt"):
                text = data.decode("utf-8", errors="ignore").replace("\x00", " ")
                text_data[chunk_type] = text_data.get(chunk_type, "") + " " + text
            i += 12 + length
    except Exception:
        pass
    return text_data


def detect(image: Image.Image, image_bytes: bytes) -> dict:
    """
    Returns:
        score: float 0.0–1.0, probability AI based on metadata analysis
        flags: list of specific evidence strings
        details: raw metadata findings
    """
    flags = []
    score_components = []
    details = {}

    # --- EXIF via exifread ---
    try:
        tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
        details["exif_tag_count"] = len(tags)

        camera_fields_present = {
            k for k in CAMERA_EXIF_FIELDS
            if any(k.lower() in str(t).lower() for t in tags.keys())
        }
        details["camera_fields"] = list(camera_fields_present)

        if len(tags) == 0:
            flags.append("No EXIF metadata — real camera photos always have EXIF")
            score_components.append(0.6)
        elif not camera_fields_present:
            flags.append("EXIF present but missing all camera-specific fields (make, model, focal length)")
            score_components.append(0.5)
        else:
            score_components.append(0.0)

        # Check software field for generator names
        software_tag = str(tags.get("Image Software", "")).lower()
        details["software"] = software_tag
        if any(sig in software_tag for sig in AI_GENERATOR_SIGNATURES):
            flags.append(f"Software EXIF field reveals AI generator: '{software_tag}'")
            score_components.append(1.0)
        else:
            score_components.append(0.0)

    except Exception as e:
        logger.warning(f"exifread error: {e}")
        score_components.extend([0.5, 0.0])

    # --- PIL EXIF cross-check ---
    try:
        exif_data = image._getexif() if hasattr(image, "_getexif") else None
        if exif_data:
            pil_tags = {TAGS.get(k, k): v for k, v in exif_data.items()}
            details["pil_software"] = str(pil_tags.get("Software", ""))
            if any(sig in details["pil_software"].lower() for sig in AI_GENERATOR_SIGNATURES):
                flags.append(f"PIL EXIF Software tag: '{details['pil_software']}'")
                score_components.append(1.0)
    except Exception:
        pass

    # --- PNG metadata chunks ---
    fmt = image.format or ""
    if fmt.upper() == "PNG":
        chunks = _read_png_text_chunks(image_bytes)
        combined_text = " ".join(chunks.values()).lower()
        details["png_text_chunks"] = combined_text[:500]

        if any(sig in combined_text for sig in AI_GENERATOR_SIGNATURES):
            flags.append("PNG tEXt chunk contains AI generator signature (prompt or tool name embedded)")
            score_components.append(1.0)
        elif "parameters" in combined_text or "prompt" in combined_text:
            flags.append("PNG tEXt chunk contains 'parameters'/'prompt' key — common in Stable Diffusion outputs")
            score_components.append(0.85)
        else:
            score_components.append(0.0)

        # No EXIF + is PNG = strong AI signal (cameras rarely save as PNG)
        if details.get("exif_tag_count", 0) == 0:
            flags.append("PNG format with no EXIF — cameras save JPEG; PNG + no metadata is typical for AI outputs")
            score_components.append(0.5)

    score = float(max(score_components)) if score_components else 0.5
    # If we found a hard generator signature, cap to 1.0
    if any("generator" in f.lower() or "signature" in f.lower() or "reveals" in f.lower() for f in flags):
        score = 1.0

    return {
        "score": round(score, 4),
        "flags": flags,
        "details": details,
    }
