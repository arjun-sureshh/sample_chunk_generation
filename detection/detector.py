"""Object detection utilities built on top of YOLOv8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    """Normalized detection payload returned by the detector."""

    bbox: list[int]
    class_name: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the detection to the requested dictionary structure."""
        return {
            "bbox": self.bbox,
            "class": self.class_name,
            "confidence": self.confidence,
        }


class YoloDetector:
    """YOLOv8 detector wrapper for retail video analytics."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.45,
        device: str = "cpu",
        image_size: int = 1280,
        target_classes: tuple[str, ...] = ("person", "cell phone", "shoe"),
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.image_size = image_size
        self.target_classes = set(target_classes)
        self.model_class_names = self._get_model_class_names()
        self.enabled_classes = self.target_classes.intersection(self.model_class_names.values())
        self.missing_classes = self.target_classes.difference(self.enabled_classes)
        self.class_ids = [
            class_id
            for class_id, class_name in self.model_class_names.items()
            if class_name in self.enabled_classes
        ]

        if self.missing_classes:
            missing = ", ".join(sorted(self.missing_classes))
            print(f"[DETECTOR] Warning: requested classes not present in model labels: {missing}")

    def _get_model_class_names(self) -> dict[int, str]:
        """Return the YOLO class-id to name mapping."""
        names = self.model.names
        if isinstance(names, dict):
            return {int(idx): str(name) for idx, name in names.items()}
        return {idx: str(name) for idx, name in enumerate(names)}

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run detection on a BGR frame and return normalized detections."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            imgsz=self.image_size,
            classes=self.class_ids or None,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        detections: list[dict[str, Any]] = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = self.model_class_names.get(class_id, str(class_id))
            if class_name not in self.enabled_classes:
                continue

            xyxy = box.xyxy[0].tolist()
            parsed = Detection(
                bbox=[int(round(coord)) for coord in xyxy],
                class_name=class_name,
                confidence=float(box.conf[0].item()),
            )
            detections.append(parsed.to_dict())

        return detections
