"""Person tracking utilities backed by ByteTrack."""

from __future__ import annotations

from collections import defaultdict, deque
import inspect
from typing import Any

import numpy as np
import supervision as sv


class PersonTracker:
    """Attach stable track IDs to person detections and store short histories."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, history_size: int = 30) -> None:
        self.history_size = history_size
        self.track_history: dict[int, deque[list[float]]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        tracker_signature = inspect.signature(sv.ByteTrack)
        tracker_kwargs: dict[str, Any] = {}
        if "lost_track_buffer" in tracker_signature.parameters:
            tracker_kwargs["lost_track_buffer"] = max_age
        if "minimum_consecutive_frames" in tracker_signature.parameters:
            tracker_kwargs["minimum_consecutive_frames"] = min_hits
        if "track_activation_threshold" in tracker_signature.parameters:
            tracker_kwargs["track_activation_threshold"] = 0.25
        self.tracker = sv.ByteTrack(**tracker_kwargs)

    def update(self, person_detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Track person detections and return them with stable track IDs."""
        if not person_detections:
            return []

        xyxy = np.array([det["bbox"] for det in person_detections], dtype=np.float32)
        confidence = np.array([det["confidence"] for det in person_detections], dtype=np.float32)
        class_id = np.zeros(len(person_detections), dtype=np.int32)

        detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        if hasattr(self.tracker, "update_with_detections"):
            tracked = self.tracker.update_with_detections(detections)
        else:
            tracked = self.tracker.update(detections)

        tracked_results: list[dict[str, Any]] = []
        tracker_ids = getattr(tracked, "tracker_id", None)
        if tracker_ids is None:
            return tracked_results

        for idx, bbox in enumerate(tracked.xyxy):
            track_id = int(tracker_ids[idx])
            parsed_bbox = [int(round(coord)) for coord in bbox.tolist()]
            center = self._compute_center(parsed_bbox)
            self.track_history[track_id].append(center)
            tracked_results.append(
                {
                    "bbox": parsed_bbox,
                    "class": "person",
                    "confidence": float(tracked.confidence[idx]),
                    "track_id": track_id,
                }
            )

        return tracked_results

    def get_track_history(self) -> dict[int, list[list[float]]]:
        """Return the recent center history for each active track."""
        return {track_id: list(history) for track_id, history in self.track_history.items()}

    @staticmethod
    def _compute_center(bbox: list[int]) -> list[float]:
        """Compute the center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
