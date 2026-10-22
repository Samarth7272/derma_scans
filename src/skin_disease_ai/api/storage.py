import json
import os
from datetime import datetime, timezone
from threading import Lock


class PredictionLogger:
    def __init__(self, path=None):
        self.path = path or os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl")
        self.backend = "jsonl"
        self.error = None
        self._lock = Lock()
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        except Exception as exc:
            self.error = str(exc)

    def log_prediction(self, request_id, filename, status, predicted_class,
                       confidence, skin_likelihood, entropy, topk):
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "filename": filename,
            "status": status,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "skin_likelihood": skin_likelihood,
            "entropy": entropy,
            "topk": [
                {"class_name": p["class_name"], "confidence": p["confidence"]}
                for p in topk
            ],
        }
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            self.error = str(exc)