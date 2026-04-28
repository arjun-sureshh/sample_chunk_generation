"""Role classification helpers for distinguishing staff from customers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from vlm.qwen_processor import classify_person


class PersonClassifier:
    """Classify tracked persons with strict LAB prefilter and VLM fallback."""

    def __init__(
        self,
        lab_config: dict[str, int],
        lab_prefilter_threshold: float = 0.92,
    ) -> None:
        self.lab_config = {
            "a_min": int(lab_config["a_min"]),
            "a_max": int(lab_config["a_max"]),
            "b_min": int(lab_config["b_min"]),
            "b_max": int(lab_config["b_max"]),
        }
        self.lab_prefilter_threshold = float(lab_prefilter_threshold)
        self.cache: dict[int, dict[str, Any]] = {}
        self.vlm_calls = 0
        self.lab_calls = 0

    def classify(self, track_id: int, person_crop_bgr: np.ndarray) -> dict[str, Any]:
        """Classify one person and cache result forever for this track id."""
        if track_id in self.cache:
            cached_result = dict(self.cache[track_id])
            cached_result["source"] = "cached"
            print(f"[CACHED] Track {track_id} → {cached_result['role']}")
            return cached_result

        if person_crop_bgr.size == 0:
            result = {
                "role": "customer",
                "confidence": 0.3,
                "source": "vlm",
                "reason": "empty crop",
            }
            self.cache[track_id] = dict(result)
            return result

        lab_score = self._check_lab(person_crop_bgr)
        if lab_score > self.lab_prefilter_threshold:
            self.lab_calls += 1
            result = {
                "role": "staff",
                "confidence": float(lab_score),
                "source": "lab",
                "reason": "uniform color match",
            }
            self.cache[track_id] = dict(result)
            print(f"[LAB] Track {track_id} → staff ({lab_score:.2f})")
            return result

        result = self._classify_via_vlm(person_crop_bgr)
        self.vlm_calls += 1
        self.cache[track_id] = dict(result)
        latency_ms = result.get("latency_ms")
        latency_suffix = f" | {int(latency_ms)}ms" if latency_ms is not None else ""
        print(
            f"[VLM] Track {track_id} → {result['role']} ({float(result['confidence']):.2f}) | "
            f"{result['reason']}{latency_suffix}"
        )
        return result

    def _check_lab(self, crop: np.ndarray) -> float:
        """Return strict LAB confidence using A/B channels and median values."""
        if crop.size == 0:
            return 0.0

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        a_value = float(np.median(lab[:, :, 1]))
        b_value = float(np.median(lab[:, :, 2]))

        a_min = float(self.lab_config["a_min"])
        a_max = float(self.lab_config["a_max"])
        b_min = float(self.lab_config["b_min"])
        b_max = float(self.lab_config["b_max"])

        a_mid = (a_min + a_max) / 2.0
        b_mid = (b_min + b_max) / 2.0
        a_half = max((a_max - a_min) / 2.0, 1.0)
        b_half = max((b_max - b_min) / 2.0, 1.0)

        a_norm = abs(a_value - a_mid) / a_half
        b_norm = abs(b_value - b_mid) / b_half
        distance = float(np.sqrt((a_norm * a_norm + b_norm * b_norm) / 2.0))
        return float(max(0.0, min(1.0, 1.0 - distance)))

    def _classify_via_vlm(self, crop: np.ndarray) -> dict[str, Any]:
        """Classify person crop via Qwen processor with guarded fallback."""
        try:
            result = classify_person(crop)
        except Exception as exc:
            print(f"[VLM ERROR] classify_person failed: {exc}")
            return {
                "role": "customer",
                "confidence": 0.3,
                "source": "vlm",
                "reason": "vlm error",
                "latency_ms": None,
            }

        role = str(result.get("role", "customer")).lower()
        if role not in {"staff", "customer", "uncertain"}:
            role = "customer"

        confidence = float(result.get("confidence", 0.3))
        confidence = max(0.0, min(1.0, confidence))
        return {
            "role": role,
            "confidence": confidence,
            "source": str(result.get("source", "vlm")),
            "reason": str(result.get("reason", "vlm classification")),
            "latency_ms": result.get("latency_ms"),
        }

    def get_cache_stats(self) -> dict[str, int]:
        """Return aggregate cache and classifier call statistics."""
        staff_count = sum(1 for item in self.cache.values() if item.get("role") == "staff")
        customer_count = sum(1 for item in self.cache.values() if item.get("role") == "customer")
        stats = {
            "total_cached": len(self.cache),
            "staff_count": staff_count,
            "customer_count": customer_count,
            "vlm_calls": self.vlm_calls,
            "lab_calls": self.lab_calls,
        }
        print(
            "[STATS] Cached: "
            f"{stats['total_cached']} | Staff: {stats['staff_count']} | Customers: {stats['customer_count']} | "
            f"VLM: {stats['vlm_calls']} | LAB: {stats['lab_calls']}"
        )
        return stats

    def clear_stale_tracks(self, active_track_ids: list[int]) -> None:
        """Remove cached tracks that are no longer active."""
        active_ids = set(active_track_ids)
        stale_ids = [track_id for track_id in self.cache if track_id not in active_ids]
        for track_id in stale_ids:
            self.cache.pop(track_id, None)


# Backward compatible alias for existing imports.
StaffCustomerClassifier = PersonClassifier
