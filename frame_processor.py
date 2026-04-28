"""Unified per-frame processing pipeline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from config import CONFIG
from detection.classifier import PersonClassifier
from detection.detector import YoloDetector
from tracking.person_state import PersonStateManager
from tracking.tracker import PersonTracker
from zones.line_counter import LineCounter
from zones.zone_engine import ZoneEngine, resolve_zone_config_path


class FrameProcessor:
    """Run detection, tracking, and role classification for one frame."""

    def __init__(self) -> None:
        detection_config = CONFIG.get("detection", {})
        tracking_config = CONFIG.get("tracking", {})
        classification_config = CONFIG.get("classification", {})
        zones_config = CONFIG.get("zones", {})
        proximity_config = CONFIG.get("proximity", {})
        footfall_config = CONFIG.get("footfall", {})
        person_state_config = CONFIG.get("person_state", {})

        self.detector = YoloDetector(
            model_path=str(detection_config.get("model", "yolov8n.pt")),
            confidence=float(detection_config.get("confidence", 0.45)),
            device=str(detection_config.get("device", "cpu")),
            image_size=int(detection_config.get("image_size", 1280)),
        )
        self.tracker = PersonTracker(
            max_age=int(tracking_config.get("max_age", 30)),
            min_hits=int(tracking_config.get("min_hits", 3)),
            history_size=30,
        )
        self.classifier = PersonClassifier(
            lab_config=dict(
                classification_config.get(
                    "staff_uniform_lab",
                    {"a_min": 110, "a_max": 150, "b_min": 90, "b_max": 155},
                )
            ),
            lab_prefilter_threshold=float(classification_config.get("lab_prefilter_threshold", 0.92)),
        )
        self.clear_stale_every_n_frames = int(classification_config.get("clear_stale_every_n_frames", 100))
        self.stats_every_n_frames = int(classification_config.get("stats_every_n_frames", 50))

        # Zones + footfall + dwell/proximity state (initialized once).
        self.zone_engine: ZoneEngine | None = None
        self.line_counter: LineCounter | None = None

        zones_path = zones_config.get("config_path")
        if zones_path:
            resolved = resolve_zone_config_path(str(zones_path))
            self.zone_engine = ZoneEngine(resolved)

        if self.zone_engine and self.zone_engine.lines:
            # Use first configured line for footfall.
            self.line_counter = LineCounter(self.zone_engine.lines[0])

        self.footfall_reset_interval_seconds = float(footfall_config.get("reset_interval_seconds", 3600))
        self.footfall_count_exits = bool(footfall_config.get("count_exits", True))
        self._last_footfall_reset_ts = time.time()

        stale_timeout = float(person_state_config.get("stale_track_timeout", 5.0))
        self.cleanup_every_n_frames = int(person_state_config.get("cleanup_every_n_frames", 100))
        self.person_state_manager = PersonStateManager(history_size=30, stale_track_timeout=stale_timeout)

        self.staff_customer_radius = float(proximity_config.get("staff_customer_radius", 150))

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> dict[str, Any]:
        """Return a structured representation for a single frame."""
        timestamp = float(timestamp)
        detections = self.detector.detect(frame)
        person_detections = [det for det in detections if det["class"] == "person"]
        object_detections = [det for det in detections if det["class"] != "person"]

        tracked_persons = self.tracker.update(person_detections)
        persons: list[dict[str, Any]] = []
        for person in tracked_persons:
            x1, y1, x2, y2 = person["bbox"]
            frame_h, frame_w = frame.shape[:2]
            x1 = max(0, min(frame_w, int(x1)))
            x2 = max(0, min(frame_w, int(x2)))
            y1 = max(0, min(frame_h, int(y1)))
            y2 = max(0, min(frame_h, int(y2)))
            crop = frame[y1:y2, x1:x2]
            role_data = self.classifier.classify(track_id=int(person["track_id"]), person_crop_bgr=crop)
            center = self._compute_center(person["bbox"])
            cx, cy = int(center[0]), int(center[1])

            # Step A: Zone assignment
            current_zones: list[str] = []
            if self.zone_engine is not None:
                current_zones = self.zone_engine.get_zones_for_point(cx, cy)

            # Step B: Person state update (zone enter/exit + dwell accumulation)
            self.person_state_manager.update(
                track_id=int(person["track_id"]),
                role=str(role_data["role"]),
                confidence=float(role_data.get("confidence", 0.0)),
                center=(cx, cy),
                current_zones=current_zones,
                timestamp=timestamp,
            )

            # Step C: Footfall line check (using the updated state history)
            footfall_event: str | None = None
            if self.line_counter is not None:
                state = self.person_state_manager.get_state(int(person["track_id"]))
                if state is not None and len(state.position_history) >= 2:
                    prev_center = state.position_history[-2]
                    curr_center = state.position_history[-1]
                    crossing = self.line_counter.update(int(person["track_id"]), prev_center, curr_center)
                    if crossing == "entry":
                        footfall_event = "entry"
                    elif crossing == "exit" and self.footfall_count_exits:
                        footfall_event = "exit"

            # Step E: enrich output with state info
            state = self.person_state_manager.get_state(int(person["track_id"]))
            staff_nearby = bool(state.staff_nearby) if state is not None else False
            unassisted_duration = (
                self.person_state_manager.get_unassisted_duration(int(person["track_id"]), timestamp)
                if role_data["role"] == "customer"
                else 0.0
            )
            dwell_times: dict[str, float] = {}
            if state is not None:
                for zone_id in state.current_zones:
                    dwell_times[zone_id] = self.person_state_manager.get_dwell_time(
                        int(person["track_id"]),
                        zone_id,
                        current_timestamp=timestamp,
                    )

            persons.append(
                {
                    "track_id": person["track_id"],
                    "role": role_data["role"],
                    "confidence": float(role_data.get("confidence", 0.0)),
                    "classification_source": str(role_data.get("source", "vlm")),
                    "reason": str(role_data.get("reason", "")),
                    "bbox": person["bbox"],
                    "center": center,
                    "current_zones": current_zones,
                    "dwell_times": dwell_times,
                    "staff_nearby": staff_nearby,
                    "unassisted_duration": float(unassisted_duration),
                    "footfall_event": footfall_event,
                }
            )

        # Step D: staff proximity check (customer -> any staff within radius)
        staff_centers: list[tuple[int, int]] = []
        for p in persons:
            if p.get("role") == "staff":
                c = p.get("center", [0, 0])
                staff_centers.append((int(c[0]), int(c[1])))

        radius_sq = float(self.staff_customer_radius * self.staff_customer_radius)
        for p in persons:
            if p.get("role") != "customer":
                continue
            c = p.get("center", [0, 0])
            cx, cy = float(c[0]), float(c[1])
            nearby = False
            for sx, sy in staff_centers:
                dx = cx - float(sx)
                dy = cy - float(sy)
                if dx * dx + dy * dy <= radius_sq:
                    nearby = True
                    break
            self.person_state_manager.update_staff_proximity(int(p["track_id"]), staff_nearby=nearby, timestamp=timestamp)
            # refresh derived values after proximity update
            p["staff_nearby"] = bool(self.person_state_manager.get_state(int(p["track_id"])).staff_nearby)  # type: ignore[union-attr]
            p["unassisted_duration"] = float(self.person_state_manager.get_unassisted_duration(int(p["track_id"]), timestamp))

        # Hourly footfall reset (configurable)
        now = time.time()
        if now - self._last_footfall_reset_ts >= self.footfall_reset_interval_seconds:
            if self.line_counter is not None:
                self.line_counter.reset_hourly()
            self._last_footfall_reset_ts = now

        if self.clear_stale_every_n_frames > 0 and frame_id % self.clear_stale_every_n_frames == 0:
            active_track_ids = [int(person["track_id"]) for person in tracked_persons]
            self.classifier.clear_stale_tracks(active_track_ids)

        # Step F: remove stale tracks periodically
        if self.cleanup_every_n_frames > 0 and frame_id % self.cleanup_every_n_frames == 0:
            active_ids = [int(person["track_id"]) for person in tracked_persons]
            self.person_state_manager.remove_stale_tracks(active_ids, timestamp)

        stats: dict[str, int] = {}
        if self.stats_every_n_frames > 0 and frame_id % self.stats_every_n_frames == 0:
            stats = self.classifier.get_cache_stats()

        # Per-zone occupancy counts (for bottom bar)
        zone_counts: dict[str, int] = {}
        for p in persons:
            for zone_id in p.get("current_zones", []) or []:
                zone_counts[zone_id] = int(zone_counts.get(zone_id, 0) + 1)

        objects = [
            {
                "class": det["class"],
                "bbox": det["bbox"],
                "confidence": det["confidence"],
            }
            for det in object_detections
        ]

        footfall_counts: dict[str, int] = {}
        if self.line_counter is not None:
            footfall_counts = self.line_counter.get_counts()

        return {
            "timestamp": float(timestamp),
            "frame_id": int(frame_id),
            "persons": persons,
            "objects": objects,
            "classification_stats": stats,
            "track_history": self.tracker.get_track_history(),
            "zone_counts": zone_counts,
            "footfall_counts": footfall_counts,
        }

    def get_track_history(self) -> dict[int, list[list[float]]]:
        """Expose recent track positions for visualization."""
        return self.tracker.get_track_history()

    @staticmethod
    def _compute_center(bbox: list[int]) -> list[int]:
        """Compute the integer center of a bounding box."""
        x1, y1, x2, y2 = bbox
        return [int((x1 + x2) / 2), int((y1 + y2) / 2)]
