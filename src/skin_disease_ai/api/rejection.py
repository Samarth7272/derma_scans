import math
from dataclasses import dataclass
from typing import Optional


def skin_likelihood_score(image) -> float:
    try:
        import numpy as np
        arr = np.asarray(image).astype("float32") / 255.0
        if arr.ndim == 3 and arr.shape[2] >= 3:
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            mask = (r > 0.35) & (g > 0.2) & (b > 0.15) & (r > b)
            return float(mask.mean())
        return 0.5
    except Exception:
        return 0.5


@dataclass
class Decision:
    is_rejected: bool
    is_normal_candidate: bool
    status: str
    reason: str
    skin_likelihood: float
    confidence: float
    margin: float
    entropy: float


def decide_prediction(
    probs,
    skin_likelihood: float = 0.0,
    min_skin_likelihood: float = 0.03,
    reject_confidence: float = 0.30,
    normal_confidence: float = 0.18,
    entropy_threshold: float = 0.93,
    enable_normal_fallback: bool = True,
) -> Decision:
    sorted_probs = sorted(probs, reverse=True)
    top1 = float(sorted_probs[0]) if len(sorted_probs) > 0 else 0.0
    top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top1 - top2
    eps = 1e-9
    entropy = -sum(float(p) * math.log(float(p) + eps) for p in probs)
    max_entropy = math.log(len(probs)) if len(probs) > 0 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    is_rejected = False
    reason = "accepted"

    if skin_likelihood < min_skin_likelihood:
        is_rejected = True
        reason = "low_skin_likelihood"
    elif top1 < reject_confidence:
        is_rejected = True
        reason = "low_confidence"
    elif norm_entropy > entropy_threshold:
        is_rejected = True
        reason = "high_entropy"

    is_normal_candidate = False
    if enable_normal_fallback and not is_rejected and top1 < normal_confidence:
        is_normal_candidate = True
        reason = "normal_candidate"

    status = "rejected" if is_rejected else ("normal_candidate" if is_normal_candidate else "accepted")

    return Decision(
        is_rejected=is_rejected,
        is_normal_candidate=is_normal_candidate,
        status=status,
        reason=reason,
        skin_likelihood=float(skin_likelihood),
        confidence=top1,
        margin=margin,
        entropy=norm_entropy,
    )
