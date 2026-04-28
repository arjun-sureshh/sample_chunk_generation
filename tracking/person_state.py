"""Per-track state tracking: zone dwell, proximity, and events."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class PersonState:
    """Mutable state for one tracked person across frames."""

    track_id: int
    role: str
    confidence: float
    current_zones: list[str] = field(default_factory=list)
    zone_entry_times: dict[str, float] = field(default_factory=dict)
    zone_dwell_times: dict[str, float] = field(default_factory=dict)
    position_history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))
    staff_nearby: bool = False
    unassisted_start: float = 0.0
    last_seen: float = 0.0
    assisted_ever: bool = False


class PersonStateManager:
    """Manage PersonState objects keyed by track id."""

    def __init__(self, history_size: int = 30, stale_track_timeout: float = 5.0) -> None:
        self.states: dict[int, PersonState] = {}
        self.history_size = int(history_size)
        self.stale_track_timeout = float(stale_track_timeout)

    def update(
        self,
        track_id: int,
        role: str,
        confidence: float,
        center: tuple[float, float] | tuple[int, int],
        current_zones: list[str],
        timestamp: float,
    ) -> None:
        """
        Update (or create) state for this person.

        Logs zone entry/exit events and accumulates dwell time.
        """

        track_id = int(track_id)
        role = str(role)
        confidence = float(confidence)
        cx, cy = float(center[0]), float(center[1])
        timestamp = float(timestamp)

        if track_id not in self.states:
            state = PersonState(
                track_id=track_id,
                role=role,
                confidence=confidence,
                position_history=deque(maxlen=self.history_size),
                staff_nearby=False,
                unassisted_start=timestamp,
                last_seen=timestamp,
            )
            self.states[track_id] = state
        else:
            state = self.states[track_id]
            state.role = role
            state.confidence = confidence

        # Update position + zones
        state.position_history.append((cx, cy))
        prev_zones = set(state.current_zones)
        new_zones = set(current_zones or [])
        state.current_zones = list(current_zones or [])

        # Zone enter events
        for zone_id in sorted(new_zones):
            if zone_id not in state.zone_entry_times:
                state.zone_entry_times[zone_id] = timestamp
                print(f"[ZONE ENTER] Track {track_id} ({role}) entered {zone_id}")

        # Zone exit events + dwell accumulation
        for zone_id in list(state.zone_entry_times.keys()):
            if zone_id not in new_zones:
                entered_at = float(state.zone_entry_times.get(zone_id, timestamp))
                dwell = max(0.0, timestamp - entered_at)
                state.zone_dwell_times[zone_id] = float(state.zone_dwell_times.get(zone_id, 0.0) + dwell)
                state.zone_entry_times.pop(zone_id, None)
                print(f"[ZONE EXIT] Track {track_id} left {zone_id} after {dwell:.1f}s")

        # If zones changed but no enter/exit (e.g. reorder), keep state stable.
        _ = prev_zones

        state.last_seen = timestamp

    def update_staff_proximity(self, customer_track_id: int, staff_nearby: bool, timestamp: float) -> None:
        """Update staff proximity and unassisted timer for a customer track."""
        customer_track_id = int(customer_track_id)
        state = self.states.get(customer_track_id)
        if state is None:
            return

        staff_nearby = bool(staff_nearby)
        timestamp = float(timestamp)
        was_nearby = bool(state.staff_nearby)
        state.staff_nearby = staff_nearby

        if staff_nearby:
            state.assisted_ever = True
            return

        # Transition: nearby -> not nearby starts unassisted timer
        if was_nearby and not staff_nearby:
            state.unassisted_start = timestamp

    def get_dwell_time(self, track_id: int, zone_id: str, current_timestamp: float | None = None) -> float:
        """
        Return total seconds spent in a zone, including ongoing dwell if currently inside.
        """
        state = self.states.get(int(track_id))
        if state is None:
            return 0.0

        zone_id = str(zone_id)
        total = float(state.zone_dwell_times.get(zone_id, 0.0))
        if zone_id in state.zone_entry_times:
            if current_timestamp is None:
                current_timestamp = float(state.last_seen)
            total += max(0.0, float(current_timestamp) - float(state.zone_entry_times[zone_id]))
        return float(total)

    def get_unassisted_duration(self, track_id: int, current_timestamp: float) -> float:
        """Return how long a customer has been without staff nearby."""
        state = self.states.get(int(track_id))
        if state is None:
            return 0.0

        if state.staff_nearby:
            return 0.0

        start = float(state.unassisted_start) if state.unassisted_start else float(state.last_seen)
        return float(max(0.0, float(current_timestamp) - start))

    def remove_stale_tracks(self, active_ids: list[int], current_timestamp: float) -> None:
        """
        Remove any tracks not active and not seen within the stale timeout.

        Logs final dwell totals before deletion.
        """
        active = {int(track_id) for track_id in (active_ids or [])}
        now = float(current_timestamp)
        stale_ids: list[int] = []
        for track_id, state in self.states.items():
            if track_id in active:
                continue
            if now - float(state.last_seen) > float(self.stale_track_timeout):
                stale_ids.append(track_id)

        for track_id in stale_ids:
            state = self.states.get(track_id)
            if state is None:
                continue
            if state.zone_dwell_times:
                dwell_parts = ", ".join(
                    f"{zone}={seconds:.1f}s" for zone, seconds in sorted(state.zone_dwell_times.items())
                )
                print(f"[DWELL FINAL] Track {track_id} ({state.role}) {dwell_parts}")
            self.states.pop(track_id, None)

    def get_state(self, track_id: int) -> PersonState | None:
        """Return state for track_id, if tracked."""
        return self.states.get(int(track_id))

    def get_all_staff_positions(self) -> list[tuple[int, tuple[float, float]]]:
        """Return (track_id, last_position) for all staff tracks."""
        results: list[tuple[int, tuple[float, float]]] = []
        for track_id, state in self.states.items():
            if state.role != "staff" or not state.position_history:
                continue
            results.append((int(track_id), state.position_history[-1]))
        return results

