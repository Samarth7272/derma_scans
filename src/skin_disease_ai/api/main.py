from __future__ import annotations

import base64
import io
import os
import uuid

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

from skin_disease_ai.config import load_config
from skin_disease_ai.api.rejection import decide_prediction, skin_likelihood_score
from skin_disease_ai.api.storage import PredictionLogger
from skin_disease_ai.data.preprocessing import build_transforms, preprocess_rgb
from skin_disease_ai.disease_info import get_disease_info
from skin_disease_ai.models.factory import load_model_from_checkpoint
from skin_disease_ai.utils.device import get_device
from skin_disease_ai.utils.io import load_checkpoint
from skin_disease_ai.xai.gradcam import GradCAM, find_last_conv_layer, overlay_heatmap

app = FastAPI(title="Skin Disease AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {"model": None, "class_names": None, "cfg": None, "device": None, "logger": None}


def _is_normal_class(class_name: str) -> bool:
    lowered = class_name.lower()
    return "normal" in lowered or "healthy" in lowered


@app.on_event("startup")
def load_model() -> None:
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "models/best.pt")
    config_path = os.getenv("CONFIG_PATH", "configs/default.yaml")
    cfg = load_config(config_path)
    device = get_device()
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model, class_to_idx = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    STATE.update(
        {
            "model": model,
            "class_names": [idx_to_class[i] for i in range(len(idx_to_class))],
            "cfg": cfg,
            "device": device,
            "logger": PredictionLogger(),
        }
    )


def _read_upload(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return np.asarray(img)


def _encode_rgb_jpeg(image_rgb: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode Grad-CAM image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


@app.get("/health")
def health():
    logger = STATE.get("logger")
    return {
        "status": "ok",
        "model_loaded": STATE["model"] is not None,
        "database_backend": getattr(logger, "backend", "disabled") if logger else "disabled",
        "database_error": getattr(logger, "error", None) if logger else None,
    }


@app.get("/classes")
def classes():
    if STATE["class_names"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"classes": STATE["class_names"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = 5, gradcam: bool = True):
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Set CHECKPOINT_PATH and restart.")
    request_id = str(uuid.uuid4())
    image = _read_upload(await file.read())
    skin_score = skin_likelihood_score(image)
    cfg = STATE["cfg"]
    processed = preprocess_rgb(
        image,
        apply_clahe=cfg.data.apply_clahe,
        remove_hair=cfg.data.remove_hair,
        denoise=cfg.data.denoise,
    )
    transform = build_transforms(cfg.data.image_size, split="test", augmentation="none")
    tensor = transform(image=processed)["image"].unsqueeze(0).to(STATE["device"])
    model = STATE["model"]
    with torch.no_grad():
        probs = model(tensor).softmax(1)[0].detach().cpu().numpy()
    indices = probs.argsort()[::-1][:topk]
    decision = decide_prediction(
        probs,
        skin_likelihood=skin_score,
        min_skin_likelihood=float(os.getenv("MIN_SKIN_LIKELIHOOD", "0.03")),
        reject_confidence=float(os.getenv("REJECT_CONFIDENCE_THRESHOLD", "0.30")),
        normal_confidence=float(os.getenv("NORMAL_CONFIDENCE_THRESHOLD", "0.18")),
        entropy_threshold=float(os.getenv("ENTROPY_REJECT_THRESHOLD", "0.93")),
        enable_normal_fallback=os.getenv("ENABLE_NORMAL_FALLBACK", "true").lower() in {"1", "true", "yes"},
    )
    predictions = [
        {
            "class_index": int(i),
            "class_name": STATE["class_names"][i],
            "confidence": float(probs[i]),
            "disease_info": get_disease_info(STATE["class_names"][i]),
        }
        for i in indices
    ]
    top_class = predictions[0]["class_name"]
    top_is_normal = _is_normal_class(top_class) and not decision.is_rejected
    status = "normal_skin" if top_is_normal else decision.status
    is_normal_candidate = bool(decision.is_normal_candidate or top_is_normal)
    predicted_class = None if decision.is_rejected else top_class
    display_label = (
        "Normal Skin"
        if top_is_normal
        else (
            "Normal / no confident abnormality detected"
            if decision.is_normal_candidate
            else ("Rejected: non-skin or uncertain image" if decision.is_rejected else top_class)
        )
    )
    response = {
        "request_id": request_id,
        "status": status,
        "is_rejected": decision.is_rejected,
        "is_normal_candidate": is_normal_candidate,
        "reason": decision.reason,
        "skin_likelihood": decision.skin_likelihood,
        "confidence": decision.confidence,
        "margin": decision.margin,
        "entropy": decision.entropy,
        "predicted_class": predicted_class,
        "display_label": display_label,
        "predictions": predictions,
    }
    logger = STATE.get("logger")
    if logger:
        logger.log_prediction(
            request_id=request_id,
            filename=file.filename,
            status=status,
            predicted_class=response["predicted_class"],
            confidence=decision.confidence,
            skin_likelihood=decision.skin_likelihood,
            entropy=decision.entropy,
            topk=predictions,
        )
    if gradcam and not decision.is_rejected and not is_normal_candidate:
        target = int(indices[0])
        try:
            cam = GradCAM(model, find_last_conv_layer(model))
            heatmap = cam(tensor, class_idx=target)
            cam.close()
            response["gradcam_jpeg_base64"] = _encode_rgb_jpeg(overlay_heatmap(image, heatmap))
        except Exception as exc:
            response["gradcam_error"] = str(exc)
    return response
