"""
Frequency domain analysis to detect AI-generated image artifacts.

AI generators (GANs, diffusion models) leave characteristic spectral fingerprints:
- GAN images: grid-like high-frequency peaks in FFT
- Diffusion images: over-smoothed mid-frequency spectrum, lack of natural 1/f noise

Strategy:
  1. Convert to grayscale, compute 2D FFT
  2. Measure spectral flatness (AI tends to be flatter / less natural)
  3. Detect grid artifacts (regular peaks indicating GAN upsampling)
  4. Measure high-frequency energy ratio
"""
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def _fft_features(gray: np.ndarray) -> dict:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_magnitude = np.log1p(magnitude)

    h, w = gray.shape
    cy, cx = h // 2, w // 2

    # High-frequency energy: outer 25% ring vs total
    mask_inner = np.zeros((h, w), dtype=bool)
    r_inner = min(h, w) // 4
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    mask_inner[dist < r_inner] = True

    total_energy = log_magnitude.sum() + 1e-9
    hf_energy = log_magnitude[~mask_inner].sum()
    hf_ratio = float(hf_energy / total_energy)

    # Spectral flatness: geometric mean / arithmetic mean of magnitude spectrum
    flat_region = magnitude[~mask_inner].flatten() + 1e-9
    geo_mean = np.exp(np.mean(np.log(flat_region)))
    arith_mean = np.mean(flat_region)
    spectral_flatness = float(geo_mean / arith_mean)  # 0–1, closer to 1 = flatter = more AI-like

    # Grid artifact detection: check for regular peaks in FFT (GAN signature)
    # Downsample magnitude for speed
    mag_small = log_magnitude[cy - 64: cy + 64, cx - 64: cx + 64]
    peak_count = int(np.sum(mag_small > np.percentile(mag_small, 99)))

    return {
        "hf_ratio": hf_ratio,
        "spectral_flatness": spectral_flatness,
        "peak_count": peak_count,
    }


def _noise_floor(gray: np.ndarray) -> float:
    """Natural images follow ~1/f noise; AI images often deviate."""
    f = np.fft.fft2(gray.astype(np.float32))
    magnitude = np.abs(np.fft.fftshift(f))
    h, w = gray.shape
    cy, cx = h // 2, w // 2

    # Radial power spectrum
    Y, X = np.ogrid[:h, :w]
    radii = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(int)
    max_r = min(cy, cx)
    radial_power = np.array([
        magnitude[radii == r].mean() if np.any(radii == r) else 0
        for r in range(1, max_r)
    ])

    # Fit log-log: natural images slope ≈ -1.0 to -1.5
    freqs = np.log(np.arange(1, len(radial_power) + 1) + 1e-9)
    power = np.log(radial_power + 1e-9)
    if len(freqs) > 2:
        slope = np.polyfit(freqs, power, 1)[0]
    else:
        slope = -1.0

    return float(slope)


def detect(image: Image.Image) -> dict:
    """
    Returns:
        score: float 0.0–1.0, probability of AI generation based on frequency analysis
        flags: list of specific anomalies detected
        details: raw feature values
    """
    try:
        gray = np.array(image.convert("L").resize((256, 256)))
        features = _fft_features(gray)
        slope = _noise_floor(gray)

        flags = []
        score_components = []

        # Spectral flatness > 0.15 is unusual for natural photos
        sf = features["spectral_flatness"]
        if sf > 0.2:
            flags.append(f"High spectral flatness ({sf:.3f}) — typical of AI generation")
            score_components.append(min(1.0, (sf - 0.15) / 0.35))
        else:
            score_components.append(0.0)

        # Power spectrum slope: natural ≈ -1.0 to -1.5; AI often shallower
        if slope > -0.7:
            flags.append(f"Shallow power spectrum slope ({slope:.2f}) — deviates from natural 1/f noise")
            score_components.append(0.7)
        elif slope > -0.9:
            score_components.append(0.4)
        else:
            score_components.append(0.0)

        # Grid artifact peaks (GAN fingerprint)
        if features["peak_count"] > 20:
            flags.append(f"Regular spectral grid peaks ({features['peak_count']}) — possible GAN upsampling artifact")
            score_components.append(0.8)
        else:
            score_components.append(0.0)

        score = float(np.mean(score_components))

        return {
            "score": round(score, 4),
            "flags": flags,
            "details": {**features, "power_slope": slope},
        }
    except Exception as e:
        logger.error(f"Frequency analyzer error: {e}")
        return {"score": 0.5, "flags": [], "details": {}, "error": str(e)}
