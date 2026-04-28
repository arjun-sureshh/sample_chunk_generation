"""Line crossing counter for entrance/exit footfall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LineCounts:
    """Aggregate footfall counts."""

    entries: int = 0
    exits: int = 0

    @property
    def net_occupancy(self) -> int:
        """Return entries - exits."""
        return int(self.entries - self.exits)


class LineCounter:
    """Count unique track crossings over a horizontal line."""

    def __init__(self, line_config: dict[str, Any]) -> None:
        """
        Initialize from a single line config dict.

        Expected keys:
        - id: str
        - start: [x, y]
        - end: [x, y]
        - direction: "down_is_entry" (currently supported)
        """

        self.line_config = dict(line_config)
        self.line_id = str(self.line_config.get("id", "line"))
        start = self.line_config.get("start", [0, 0])
        end = self.line_config.get("end", [0, 0])
        self.line_y = float(start[1]) if start and len(start) > 1 else 0.0
        if end and len(end) > 1:
            # If provided line isn't perfectly horizontal, we still treat crossing by avg Y.
            self.line_y = float(start[1] + end[1]) / 2.0

        self.direction = str(self.line_config.get("direction", "down_is_entry"))
        self.counts = LineCounts()
        self.crossed_track_ids: set[int] = set()

    def update(
        self,
        track_id: int,
        prev_center: tuple[float, float] | tuple[int, int],
        curr_center: tuple[float, float] | tuple[int, int],
    ) -> str | None:
        """
        Update counters based on a track's previous and current centers.

        Returns:
        - "entry" if crossed downward across the line
        - "exit" if crossed upward across the line
        - None if no crossing or already counted
        """

        track_id = int(track_id)
        if track_id in self.crossed_track_ids:
            return None

        prev_cy = float(prev_center[1])
        curr_cy = float(curr_center[1])
        y = float(self.line_y)

        crossed_down = prev_cy < y and curr_cy >= y
        crossed_up = prev_cy > y and curr_cy <= y

        if crossed_down:
            self.counts.entries += 1
            self.crossed_track_ids.add(track_id)
            print(f"[FOOTFALL] Entry! Total entries: {self.counts.entries}")
            return "entry"

        if crossed_up:
            self.counts.exits += 1
            self.crossed_track_ids.add(track_id)
            print(f"[FOOTFALL] Exit! Total exits: {self.counts.exits}")
            return "exit"

        return None

    def get_counts(self) -> dict[str, int]:
        """Return current entry/exit counts and net occupancy."""
        return {
            "entries": int(self.counts.entries),
            "exits": int(self.counts.exits),
            "net_occupancy": int(self.counts.net_occupancy),
        }

    def reset_hourly(self) -> None:
        """Reset counts and clear the de-duplication set."""
        self.counts = LineCounts()
        self.crossed_track_ids.clear()
        print("[FOOTFALL] Hourly reset done")

