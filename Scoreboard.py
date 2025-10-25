import os
from datetime import datetime
from typing import Iterable, List


TABLE_HEADERS = ["End", "Arrow 1", "Arrow 2", "Arrow 3", "Arrow 4", "Arrow 5", "Total"]
TABLE_WIDTHS = [5, 9, 9, 9, 9, 9, 7]


def _ensure_output_dir(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _format_minutes_seconds(seconds: float) -> str:
    total_seconds = int(round(max(float(seconds or 0.0), 0.0)))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes}m {secs:02d}s"


def _table_border() -> str:
    return "+" + "+".join("-" * width for width in TABLE_WIDTHS) + "+"


def _format_row(values: List[str]) -> str:
    cells = [f" {value:<{width - 1}}" for value, width in zip(values, TABLE_WIDTHS)]
    return "|" + "|".join(cells) + "|"


def _chunk_scores(scores: List[int], chunk_size: int = 5) -> Iterable[List[int]]:
    for idx in range(0, len(scores), chunk_size):
        yield scores[idx : idx + chunk_size]


def _build_table_lines(scores: List[int]) -> List[str]:
    lines = [_table_border(), _format_row(TABLE_HEADERS), _table_border()]

    if not scores:
        empty_row = ["1"] + ["-"] * 5 + ["0"]
        lines.append(_format_row(empty_row))
        lines.append(_table_border())
        return lines

    for end_index, chunk in enumerate(_chunk_scores(scores), start=1):
        row = [str(end_index)]
        total = 0
        for arrow_idx in range(5):
            if arrow_idx < len(chunk):
                score = chunk[arrow_idx]
                row.append(str(score))
                total += score
            else:
                row.append("-")
        row.append(str(total))
        lines.append(_format_row(row))

    lines.append(_table_border())
    return lines


def write_run_report(
    *,
    run_started_at: datetime,
    run_finished_at: datetime,
    run_duration: float,
    telemetry: dict,
    verified_hits: List,
    output_dir: str,
) -> str:
    """Write a scoreboard summary for a completed analysis run."""

    if run_finished_at is None:
        run_finished_at = datetime.now()
    if run_started_at is None:
        run_started_at = run_finished_at

    output_dir = _ensure_output_dir(output_dir)
    timestamp_for_file = run_finished_at.strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join(output_dir, f"scoreboard_{timestamp_for_file}.txt")

    verified_scores = [getattr(hit, "score", 0) for hit in verified_hits]
    verified_total = sum(verified_scores)
    run_duration = float(run_duration or 0.0)
    component_total = 0.0
    run_clock = run_duration
    if telemetry:
        component_keys = ["target_detection", "radial_filtering", "hit_scoring", "bookkeeping"]
        component_total = sum(float(telemetry.get(k, 0.0)) for k in component_keys)
        run_clock = float(telemetry.get("run_clock", run_duration))
    variance_percent = 0.0
    if run_clock > 0 and telemetry:
        variance_percent = ((component_total - run_clock) / run_clock) * 100.0
    run_formatted = _format_minutes_seconds(run_clock)

    lines: List[str] = []
    lines.append(f"Run timestamp: {run_started_at.isoformat(timespec='seconds')}")
    if telemetry:
        lines.append(f"Run duration: {run_formatted} ({variance_percent:+.1f}% variance)")
    else:
        lines.append(f"Run duration: {run_formatted}")
    lines.append(f"Verified hits: {len(verified_scores)}")
    lines.append(f"Grand total: {verified_total}")
    lines.append("")
    lines.append("Telemetry (seconds):")

    telemetry_labels = [
        ("target_detection", "Target Detection"),
        ("radial_filtering", "Radial Filtering"),
        ("radial_warp_subtract", "    Radial: Warp + Subtract"),
        ("radial_distance_map", "    Radial: Distance Map"),
        ("radial_ring_estimate", "    Radial: Ring Estimate"),
        ("radial_threshold_morph", "    Radial: Threshold/Morph"),
        ("radial_line_detect", "    Radial: Line Detect"),
        ("radial_contour_reconstruct", "    Radial: Contour Reconstruct"),
        ("hit_scoring", "Hit Scoring"),
        ("bookkeeping", "Bookkeeping"),
        ("bookkeeping_hits", "    Bookkeeping: Hit Maintenance"),
        ("bookkeeping_grouping", "    Bookkeeping: Grouping"),
        ("bookkeeping_overlay", "    Bookkeeping: Overlay/Draw"),
        ("bookkeeping_io", "    Bookkeeping: I/O (Resize/Display/Write)"),
        ("bookkeeping_idle", "    Bookkeeping: Idle/Unaccounted"),
    ]
    for key, label in telemetry_labels:
        value = float(telemetry.get(key, 0.0)) if telemetry else 0.0
        lines.append(f"  - {label}: {_format_minutes_seconds(value)}")

    lines.append("")
    lines.append("Ends (5 arrows per end):")
    lines.extend(_build_table_lines(verified_scores))

    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return file_path
