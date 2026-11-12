#!/usr/bin/env python3
"""Forecast Roblox Up-and-Coming chart ranks using only recorded chart history."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Historical walk-forward tuning on the public archive favors mostly retaining
# probability on games already visible, while widening rank uncertainty over time.
NAMED_MASS_START = 0.90
NAMED_MASS_END = 0.84
BASE_TEMPERATURE = 2.0
TEMPERATURE_GROWTH = 0.70
TREND_SHRINK = 0.70
TREND_LIMIT_RANKS_PER_HOUR = 0.25
MICRO = 1_000_000


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def load_history(path: Path) -> list[dict[str, Any]]:
    marks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                mark = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid history JSON on line {line_number}") from exc
            if mark.get("ok") is True and isinstance(mark.get("up_and_coming"), list):
                rows = [r for r in mark["up_and_coming"]
                        if isinstance(r, dict) and isinstance(r.get("universe_id"), int)
                        and isinstance(r.get("rank"), int)]
                if rows:
                    mark = dict(mark)
                    mark["up_and_coming"] = rows
                    marks.append(mark)
    if not marks:
        raise ValueError("history contains no successful chart reads")
    def sort_key(mark: dict[str, Any]) -> float:
        when = parse_time(mark.get("at"))
        return when.timestamp() if when is not None else float("-inf")

    marks.sort(key=sort_key)
    return marks


def rank_velocity(universe_id: int, current_time: datetime | None,
                  marks: list[dict[str, Any]], latest_index: int) -> float:
    """Robust, recency-weighted rank slope in places per hour."""
    observations: list[tuple[float, float]] = []
    lower = max(0, latest_index - 12)
    for mark in marks[lower:latest_index + 1]:
        when = parse_time(mark.get("at"))
        if when is None or current_time is None:
            continue
        age = (when - current_time).total_seconds() / 3600.0
        row = next((r for r in mark["up_and_coming"] if r["universe_id"] == universe_id), None)
        if row is not None:
            observations.append((age, float(row["rank"])))
    if len(observations) < 2:
        return 0.0

    # Weighted linear regression; recent observations matter most, and irregular
    # archive intervals are handled through their timestamps rather than indices.
    weights = [math.exp(age / 18.0) for age, _ in observations]
    weight_sum = sum(weights)
    mean_x = sum(w * x for w, (x, _) in zip(weights, observations)) / weight_sum
    mean_y = sum(w * y for w, (_, y) in zip(weights, observations)) / weight_sum
    variance = sum(w * (x - mean_x) ** 2 for w, (x, _) in zip(weights, observations))
    if variance <= 1e-9:
        return 0.0
    covariance = sum(w * (x - mean_x) * (y - mean_y)
                     for w, (x, y) in zip(weights, observations))
    slope = covariance / variance
    return max(-TREND_LIMIT_RANKS_PER_HOUR, min(TREND_LIMIT_RANKS_PER_HOUR, slope))


def forecast(task: dict[str, Any], marks: list[dict[str, Any]]) -> dict[str, Any]:
    places = int(task.get("places", 10))
    if places < 1:
        raise ValueError("task places must be positive")

    latest_index = len(marks) - 1
    latest = marks[latest_index]
    current_time = parse_time(latest.get("at")) or parse_time(task.get("history_until"))
    rows = sorted(latest["up_and_coming"], key=lambda r: (r["rank"], r["universe_id"]))
    # De-duplicate malformed snapshots deterministically.
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_id.setdefault(row["universe_id"], row)
    rows = list(by_id.values())
    if not rows:
        raise ValueError("latest successful chart read has no games")

    horizon = int((task.get("window") or {}).get("slots", 72))
    horizon = max(1, horizon)
    slot_minutes = float((task.get("window") or {}).get("slot_minutes", 60))
    cadence_hours = max(1.0 / 60.0, slot_minutes / 60.0)
    ids = [row["universe_id"] for row in rows]
    current_ranks = {row["universe_id"]: float(row["rank"]) for row in rows}
    velocities = {uid: rank_velocity(uid, current_time, marks, latest_index) for uid in ids}

    # Average each game's place distribution across the actual hourly window.
    # Unnamed mass is crucial: future-created games have no name in history.
    accumulated = {uid: [0.0] * places for uid in ids}
    for hour_index in range(horizon):
        hours = hour_index * cadence_hours
        temperature = BASE_TEMPERATURE + TEMPERATURE_GROWTH * math.sqrt(hours)
        named_mass = NAMED_MASS_START + (NAMED_MASS_END - NAMED_MASS_START) * (
            hour_index / max(1, horizon - 1)
        )
        centers: dict[int, float] = {}
        for uid in ids:
            # Trend is capped to avoid extrapolating noisy chart jumps indefinitely.
            projected = current_ranks[uid] + velocities[uid] * TREND_SHRINK * hours
            centers[uid] = max(1.0, min(110.0, projected))
        for place_index in range(places):
            target_rank = place_index + 1
            weights = [math.exp(-abs(centers[uid] - target_rank) / temperature) for uid in ids]
            denominator = sum(weights)
            if denominator <= 0.0:
                continue
            for uid, weight in zip(ids, weights):
                accumulated[uid][place_index] += (
                    named_mass * weight / denominator / horizon
                )

    probabilities: list[dict[str, Any]] = []
    for place_index in range(places):
        # Normalize down after integer rounding, never above one. The residual is
        # intentionally distributed by the scorer over games we did not name.
        emitted = []
        for uid in ids:
            p = math.floor(accumulated[uid][place_index] * MICRO) / MICRO
            if p > 0:
                emitted.append((uid, p))
        emitted.sort(key=lambda item: (-item[1], item[0]))
        # The kit allows at most 100 rows at each place.
        emitted = emitted[:100]
        for uid, p in emitted:
            probabilities.append({"universe_id": uid, "place": place_index + 1, "p": p})

    # The tie-break uses the expected rank averaged over the same forecast window.
    avg_ranks = []
    for uid in ids:
        mean_rank = 0.0
        for hour_index in range(horizon):
            hours = hour_index * cadence_hours
            mean_rank += max(1.0, min(110.0, current_ranks[uid] +
                                velocities[uid] * TREND_SHRINK * hours)) / horizon
        avg_ranks.append((mean_rank, current_ranks[uid], uid))
    top10 = [uid for _, _, uid in sorted(avg_ranks)[:places]]
    if len(top10) < places:
        raise ValueError("chart has fewer distinct games than scored places")
    return {"probabilities": probabilities, "top10": top10}


def main() -> None:
    task = json.loads((INPUT_DIR / "task.json").read_text(encoding="utf-8"))
    marks = load_history(INPUT_DIR / "history.ndjson")
    result = forecast(task, marks)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = task.get("output", "forecast.json")
    (OUTPUT_DIR / output_name).write_text(
        json.dumps(result, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
