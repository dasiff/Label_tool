from __future__ import annotations
import base64
import io
from PIL import Image
import numpy as np


def decode_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an OpenCV BGR uint8 array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    bgr = rgb[:, :, ::-1].copy()
    return bgr


def encode_bgr_to_base64_png(bgr: np.ndarray) -> str:
    """Encode an OpenCV BGR uint8 image as base64 PNG."""
    from PIL import Image
    rgb = bgr[:, :, ::-1]
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
