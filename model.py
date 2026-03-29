from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
import onnxruntime as ort
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resnet50_extractor.onnx"
KNN_PATH = BASE_DIR / "knn_index.pkl"

# PatchCore settings.
INPUT_SIZE = (224, 224)
PATCH_GRID = (28, 28)
BEST_THRESHOLD = 20.347673
PIXEL_THRESHOLD_FACTOR = 1.0


@lru_cache(maxsize=1)
def _load_runtime() -> tuple[ort.InferenceSession, object, str, str]:
    """Load ONNX session and KNN index once and reuse across predict calls."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing ONNX model: {MODEL_PATH}")
    if not KNN_PATH.exists():
        raise FileNotFoundError(f"Missing KNN index: {KNN_PATH}")

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    with KNN_PATH.open("rb") as f:
        nn_index = pickle.load(f)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, nn_index, input_name, output_name


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    """Accept CHW/HWC input and return HWC uint8 RGB."""
    arr = np.asarray(image)

    if arr.ndim != 3:
        raise ValueError("Input image must have 3 dimensions.")

    # Evaluator provides 3x1024x1024. Also support HxWx3 for local testing.
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            # Support both [0, 1] and [0, 255] float inputs.
            scale = 255.0 if arr.max() <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def _preprocess_from_array(image_hwc_uint8: np.ndarray) -> np.ndarray:
    """Convert HWC uint8 image to PatchCore ONNX input NCHW float32."""
    img = Image.fromarray(image_hwc_uint8, mode="RGB").resize(INPUT_SIZE, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def _resize_mask_to_original(mask_small: np.ndarray, height: int, width: int) -> np.ndarray:
    """Upsample binary mask from patch resolution to original image size."""
    pil_mask = Image.fromarray(mask_small.astype(np.uint8) * 255, mode="L")
    pil_mask = pil_mask.resize((width, height), Image.NEAREST)
    return np.asarray(pil_mask, dtype=np.uint8)


def predict(image: np.ndarray) -> np.ndarray:
    """PatchCore segmentation used by evaluator.

    Args:
        image: input image in CHW (3, H, W) or HWC (H, W, 3).

    Returns:
        Binary mask with shape (H, W), uint8 values {0, 255}.
    """
    image_hwc = _to_hwc_uint8(image)
    h, w, _ = image_hwc.shape

    session, nn_index, input_name, output_name = _load_runtime()
    arr = _preprocess_from_array(image_hwc)

    features = session.run([output_name], {input_name: arr})[0].astype(np.float32)
    distances, _ = nn_index.kneighbors(features, n_neighbors=1, return_distance=True)
    dist_score = distances[:, 0]

    # Patch-level anomaly map and threshold to binary segmentation.
    segm_map = dist_score.reshape(PATCH_GRID)
    mask_small = segm_map >= (BEST_THRESHOLD * PIXEL_THRESHOLD_FACTOR)
    mask = _resize_mask_to_original(mask_small, h, w)

    return mask

