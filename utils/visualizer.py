"""Visualization helpers for annotated retail analytics frames."""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from config import CONFIG
from zones.zone_engine import ZoneEngine, resolve_zone_config_path


class FrameVisualizer:
    """Draw tracked entities and save annotated frames to disk."""

    def __init__(self, output_dir: str, zone_engine: ZoneEngine | None = None) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        zones_config = CONFIG.get("zones", {})
        self.draw_zones_enabled = bool(zones_config.get("draw_on_frame", False))
        self.zone_opacity = float(zones_config.get("zone_opacity", 0.15))

        # Prefer the single ZoneEngine instance created by FrameProcessor.
        self.zone_engine: ZoneEngine | None = zone_engine
        if self.zone_engine is None:
            zones_path = zones_config.get("config_path")
            if zones_path:
                try:
                    resolved = resolve_zone_config_path(str(zones_path))
                    self.zone_engine = ZoneEngine(resolved)
                except Exception as exc:
                    print(f"[ZONE] Failed to load zone config for visualization: {exc}")
                    self.zone_engine = None

    def annotate(self, frame: np.ndarray, frame_data: dict[str, Any]) -> np.ndarray:
        """Return a copy of the frame with overlays applied."""
        annotated = frame.copy()

        # Step 1: Draw zones behind person boxes.
        if self.draw_zones_enabled and self.zone_engine is not None:
            annotated = self.zone_engine.draw_zones(annotated, alpha=self.zone_opacity)

        track_history = frame_data.get("track_history", {})
        stats = frame_data.get("classification_stats", {})
        zone_counts: dict[str, int] = dict(frame_data.get("zone_counts", {}) or {})
        footfall_counts: dict[str, int] = dict(frame_data.get("footfall_counts", {}) or {})
        staff_count = 0
        customer_count = 0

        for person in frame_data.get("persons", []):
            role = person["role"]
            confidence = float(person.get("confidence", 0.0))
            source = str(person.get("classification_source", "vlm"))
            unassisted_duration = float(person.get("unassisted_duration", 0.0) or 0.0)
            staff_nearby = bool(person.get("staff_nearby", False))

            if role == "staff":
                color = (0, 255, 0)
                staff_count += 1
            elif role == "customer":
                # Step 4: highlight unassisted customers
                if unassisted_duration > 30.0 and not staff_nearby:
                    color = (0, 165, 255)  # orange (BGR)
                else:
                    color = (255, 0, 0)
                customer_count += 1
            else:
                color = (0, 255, 255)
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            role_text = role.upper()
            if role in {"staff", "customer"}:
                role_text = f"{role_text}({source}) {confidence:.2f}"
            else:
                role_text = f"{role_text} {confidence:.2f}"
            label = f"ID {person['track_id']} | {role_text}"
            if role == "customer" and unassisted_duration > 30.0 and not staff_nearby:
                label = f"{label} ⚠️ {int(unassisted_duration)}s unassisted"
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # Step 2: show zone info under bbox label
            zone_ids = person.get("current_zones", []) or []
            if zone_ids:
                zone_label = str(zone_ids[0])
                if self.zone_engine is not None:
                    zone_label = self.zone_engine.get_zone_label(str(zone_ids[0]))
                cv2.putText(
                    annotated,
                    f"📍 {zone_label}",
                    (x1, min(y2 + 20, annotated.shape[0] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            history = track_history.get(person["track_id"], [])
            if len(history) > 1:
                points = np.array(
                    [[int(point[0]), int(point[1])] for point in history],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))
                cv2.polylines(annotated, [points], isClosed=False, color=color, thickness=2)

        for obj in frame_data.get("objects", []):
            if obj["class"] != "cell phone":
                continue
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                "cell phone",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        vlm_calls = int(stats.get("vlm_calls", 0))
        lab_calls = int(stats.get("lab_calls", 0))
        summary_text = f"Staff: {staff_count} | Customers: {customer_count} | VLM calls: {vlm_calls} | LAB calls: {lab_calls}"
        cv2.putText(
            annotated,
            summary_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Step 3: bottom bar stats (footfall + key zones)
        entries = int(footfall_counts.get("entries", 0))
        exits = int(footfall_counts.get("exits", 0))
        bottom_parts = [f"Footfall Today: {entries} IN | {exits} OUT"]
        if self.zone_engine is not None and zone_counts:
            # show a couple of configured zones in a stable order
            for zone in self.zone_engine.get_all_zones()[:4]:
                zid = str(zone.get("id"))
                label = self.zone_engine.get_zone_label(zid)
                count = int(zone_counts.get(zid, 0))
                bottom_parts.append(f"{label}: {count} persons")
        bottom_text = " | ".join(bottom_parts)
        y = annotated.shape[0] - 15
        cv2.putText(
            annotated,
            bottom_text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return annotated

    def save(self, frame: np.ndarray, relative_path: str) -> str:
        """Save an annotated frame and return the output path."""
        output_path = os.path.join(self.output_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)
        return output_path

    def save_video_frame(self, writer: cv2.VideoWriter, frame: np.ndarray) -> None:
        """Write one annotated frame into an open video writer."""
        writer.write(frame)
