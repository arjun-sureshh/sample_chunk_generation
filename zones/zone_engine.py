"""Polygon zone engine and line crossing helper."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import cv2
import numpy as np


class ZoneEngine:
    """Load zone/line config and provide zone queries + drawing utilities."""

    def __init__(self, zone_config_path: str) -> None:
        """
        Load JSON config file and parse all zones and lines.

        Args:
            zone_config_path: Path to a JSON file describing zones and lines.
        """

        self.zone_config_path = str(zone_config_path)
        with open(self.zone_config_path, "r", encoding="utf-8") as f:
            self.config: dict[str, Any] = json.load(f)

        self.camera_id = str(self.config.get("camera_id", "unknown"))
        self.resolution = tuple(self.config.get("resolution", [0, 0]))

        self.zones: dict[str, dict[str, Any]] = {}
        self._zone_polygons: dict[str, np.ndarray] = {}
        self._zone_labels: dict[str, str] = {}
        self._zone_usecases: dict[str, set[str]] = {}

        zones = self.config.get("zones", []) or []
        for zone in zones:
            zone_id = str(zone["id"])
            self.zones[zone_id] = dict(zone)
            self._zone_labels[zone_id] = str(zone.get("label", zone_id))
            self._zone_usecases[zone_id] = set(zone.get("use_cases", []) or [])

            points = zone.get("points", []) or []
            poly = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            self._zone_polygons[zone_id] = poly

        self.lines: list[dict[str, Any]] = [dict(item) for item in (self.config.get("lines", []) or [])]

        print(f"[ZONE] Loaded {len(self.zones)} zones and {len(self.lines)} lines from {self.camera_id}")

        self._crossed_track_ids: set[int] = set()
        self._cross_reset_seconds = 3600
        self._last_cross_reset_ts = time.time()

    def set_cross_reset_interval_seconds(self, seconds: float) -> None:
        """Configure how often to reset internal crossing de-duplication state."""
        self._cross_reset_seconds = max(float(seconds), 1.0)

    def _maybe_reset_cross_set(self, timestamp: float) -> None:
        if timestamp - self._last_cross_reset_ts >= self._cross_reset_seconds:
            self._crossed_track_ids.clear()
            self._last_cross_reset_ts = float(timestamp)

    def get_zones_for_point(self, cx: int, cy: int) -> list[str]:
        """
        Return list of zone IDs containing the point.

        Uses cv2.pointPolygonTest: >= 0 means inside/on edge.
        """

        point = (int(cx), int(cy))
        hits: list[str] = []
        for zone_id, polygon in self._zone_polygons.items():
            # polygon is shape (N,1,2) int32
            inside = cv2.pointPolygonTest(polygon, point, False)
            if inside >= 0:
                hits.append(zone_id)
        return hits

    def get_zone_label(self, zone_id: str) -> str:
        """Return the human readable label for a zone id."""
        return str(self._zone_labels.get(str(zone_id), zone_id))

    def get_zones_by_usecase(self, use_case: str) -> list[str]:
        """Return all zone IDs that include the given use case."""
        use_case = str(use_case)
        return [zone_id for zone_id, use_cases in self._zone_usecases.items() if use_case in use_cases]

    def get_all_zones(self) -> list[dict[str, Any]]:
        """Return list of all zone config dicts."""
        return [dict(zone) for zone in self.zones.values()]

    def draw_zones(self, frame: np.ndarray, alpha: float = 0.15) -> np.ndarray:
        """
        Draw polygon zones and lines on a copy of the given frame.

        - Semi-transparent fill for polygons
        - Border + label
        - Entrance line + label
        """

        annotated = frame.copy()
        overlay = annotated.copy()

        for zone_id, zone in self.zones.items():
            color = tuple(int(c) for c in (zone.get("color", [0, 255, 0]) or [0, 255, 0]))
            polygon = self._zone_polygons[zone_id]

            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(annotated, [polygon], isClosed=True, color=color, thickness=2)

            # label near top-left of polygon bbox
            xs = polygon[:, 0, 0]
            ys = polygon[:, 0, 1]
            x_min = int(xs.min()) if xs.size else 0
            y_min = int(ys.min()) if ys.size else 0
            label = self.get_zone_label(zone_id)
            cv2.putText(
                annotated,
                label,
                (x_min + 5, max(15, y_min + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        annotated = cv2.addWeighted(overlay, float(alpha), annotated, float(1.0 - alpha), 0.0)

        for line in self.lines:
            if str(line.get("type", "line")) != "line":
                continue
            start = line.get("start", [0, 0])
            end = line.get("end", [0, 0])
            color = tuple(int(c) for c in (line.get("color", [0, 255, 255]) or [0, 255, 255]))
            p1 = (int(start[0]), int(start[1]))
            p2 = (int(end[0]), int(end[1]))
            cv2.line(annotated, p1, p2, color, 3)
            cv2.putText(
                annotated,
                "ENTRANCE",
                (min(p1[0], p2[0]) + 10, int((p1[1] + p2[1]) / 2) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return annotated

    def get_line_crossing(self, track_id: int, prev_cy: float, curr_cy: float, timestamp: float | None = None) -> str | None:
        """
        Check if person crossed the configured entrance line.

        Logic:
        - If prev_cy < line_y and curr_cy >= line_y => "entry"
        - If prev_cy > line_y and curr_cy <= line_y => "exit"
        - De-dupe per track_id until reset interval elapses.
        """

        if not self.lines:
            return None

        if timestamp is None:
            timestamp = time.time()

        self._maybe_reset_cross_set(float(timestamp))

        track_id = int(track_id)
        if track_id in self._crossed_track_ids:
            return None

        entrance = self.lines[0]
        start = entrance.get("start", [0, 0])
        end = entrance.get("end", [0, 0])
        line_y = float(start[1] + end[1]) / 2.0 if len(end) > 1 else float(start[1])

        if float(prev_cy) < line_y and float(curr_cy) >= line_y:
            self._crossed_track_ids.add(track_id)
            return "entry"
        if float(prev_cy) > line_y and float(curr_cy) <= line_y:
            self._crossed_track_ids.add(track_id)
            return "exit"
        return None


def resolve_zone_config_path(config_path: str) -> str:
    """
    Resolve a zone config path relative to current working directory.

    This keeps behavior stable when running from `sample_chunk_generation/`.
    """

    config_path = str(config_path)
    if os.path.isabs(config_path):
        return config_path
    return os.path.join(os.getcwd(), config_path)

