from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ChartPoint = tuple[str, float, float, float, float, float, float]
PlacedChartPoint = tuple[ChartPoint, float, float]

PREFERRED_EXAMPLE_STRATEGIES = (
    "baseline",
    "uniform",
    "acoustic_salience",
    "energy_salience",
    "energy_tuned",
    "patch",
)


def write_publication_artifacts(output_dir: Path) -> dict:
    from audiotokenlab.listening_study import write_listening_study_artifacts
    from audiotokenlab.serving import write_serving_stack_report

    metrics_rows = _read_csv_dicts(output_dir / "metrics.csv")
    asr_rows = _read_csv_dicts(output_dir / "asr_metrics.csv")
    speaker_rows = _read_csv_dicts(output_dir / "speaker_metrics.csv")
    summary = build_publication_summary(metrics_rows, asr_rows, speaker_rows)
    summary["listening_study"] = write_listening_study_artifacts(output_dir)
    summary["serving_stack"] = write_serving_stack_report(output_dir)
    (output_dir / "publication_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary_chart(output_dir / "summary_chart.svg", summary["strategy_summary"])
    write_listening_examples(output_dir / "listening_examples.md", asr_rows)
    return summary


def build_publication_summary(
    metrics_rows: list[dict],
    asr_rows: list[dict],
    speaker_rows: list[dict],
) -> dict:
    strategies = sorted(
        {
            str(row["strategy"])
            for row in metrics_rows + asr_rows + speaker_rows
            if row.get("strategy")
        }
    )
    summary: dict[str, dict] = {}
    for strategy in strategies:
        metric_group = [row for row in metrics_rows if row.get("strategy") == strategy]
        asr_group = [row for row in asr_rows if row.get("strategy") == strategy]
        speaker_group = [row for row in speaker_rows if row.get("strategy") == strategy]
        summary[strategy] = {
            "row_count": max(len(metric_group), len(asr_group), len(speaker_group)),
            "mean_token_reduction_ratio": _mean(metric_group, "token_reduction_ratio"),
            "mean_wer": _mean(asr_group, "wer"),
            "mean_cer": _mean(asr_group, "cer"),
            "mean_speaker_similarity": _mean(speaker_group, "speaker_similarity"),
            "mean_kv_cache_savings_mb": _mean(
                metric_group,
                "estimated_kv_cache_savings_mb",
            ),
            "mean_reconstruction_snr_db": _mean(metric_group, "reconstruction_snr_db"),
        }

    best_energy = _best_energy_strategy(summary)
    return {
        "strategy_count": len(strategies),
        "strategy_summary": summary,
        "best_energy_strategy": best_energy,
    }


def write_summary_chart(path: Path, summary: dict[str, dict]) -> None:
    width = 1040
    height = 620
    margin_left = 90
    margin_right = 380
    margin_top = 64
    margin_bottom = 78
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    points = []
    for strategy, values in summary.items():
        reduction = float(values.get("mean_token_reduction_ratio", 0.0))
        wer = float(values.get("mean_wer", 0.0))
        speaker = float(values.get("mean_speaker_similarity", 0.0))
        x = margin_left + reduction * plot_width
        y = margin_top + min(1.0, wer) * plot_height
        radius = 7 + max(0.0, min(1.0, speaker)) * 15
        points.append((strategy, x, y, radius, reduction, wer, speaker))

    palette = {
        "baseline": "#2f6fbb",
        "uniform": "#b2552d",
        "acoustic_salience": "#2f8f5b",
        "energy_salience": "#7661b3",
        "energy_tuned": "#8a7a25",
        "patch": "#8b2f3f",
    }
    points = _ordered_chart_points(points)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>AudioTokenLab token reduction vs WER</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="30" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#172026">Token reduction vs ASR WER</text>',
        '<text x="40" y="52" font-family="Arial, sans-serif" font-size="13" fill="#53606b">Bubble size is mean speaker similarity. Lower WER is better. Close bubbles are offset with leader lines.</text>',
        _line(margin_left, margin_top, margin_left, margin_top + plot_height),
        _line(margin_left, margin_top + plot_height, margin_left + plot_width, margin_top + plot_height),
    ]
    for tick in range(0, 6):
        ratio = tick / 5
        x = margin_left + ratio * plot_width
        y = margin_top + plot_height
        elements.append(_line(x, y, x, y + 6))
        elements.append(_text(x - 16, y + 28, f"{ratio:.0%}", 12))
    for tick in range(0, 6):
        ratio = tick / 5
        x = margin_left
        y = margin_top + ratio * plot_height
        elements.append(_line(x - 6, y, x, y))
        elements.append(_text(30, y + 4, f"{ratio:.0%}", 12))
    elements.append(_text(margin_left + 170, height - 22, "Mean token reduction", 13))
    elements.append(
        f'<text x="20" y="{margin_top + 200}" font-family="Arial, sans-serif" font-size="13" fill="#53606b" transform="rotate(-90 20 {margin_top + 200})">Mean WER</text>'
    )

    display_points = _spread_chart_points(
        points,
        margin_left,
        margin_left + plot_width,
        margin_top,
        margin_top + plot_height,
    )
    for (
        _index,
        ((_, x, y, _radius, _reduction, _wer, _speaker), display_x, display_y),
    ) in enumerate(display_points, 1):
        if abs(display_x - x) > 1.0 or abs(display_y - y) > 1.0:
            elements.append(_line(x, y, display_x, display_y, stroke="#c8d0d8", dash="3 3"))
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="#ffffff" stroke="#53606b" stroke-width="1"/>'
            )

    for (
        index,
        ((strategy, _x, _y, radius, _reduction, _wer, _speaker), display_x, display_y),
    ) in enumerate(display_points, 1):
        color = _strategy_color(strategy, palette)
        elements.append(
            f'<circle cx="{display_x:.1f}" cy="{display_y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.78" stroke="#172026" stroke-width="1"/>'
        )
        elements.append(
            _text(
                display_x - (4 if index < 10 else 7),
                display_y + 4,
                str(index),
                13,
                fill="#ffffff",
                weight="700",
            )
        )
    legend_x = margin_left + plot_width + 36
    legend_y = margin_top
    elements.append(_text(legend_x, legend_y - 14, "Legend", 15, weight="700"))
    for index, (strategy, _x, _y, _radius, reduction, wer, speaker) in enumerate(points, 1):
        color = _strategy_color(strategy, palette)
        y = legend_y + (index - 1) * 34
        elements.append(
            f'<circle cx="{legend_x + 9:.1f}" cy="{y - 4:.1f}" r="9" fill="{color}" fill-opacity="0.86" stroke="#172026" stroke-width="1"/>'
        )
        elements.append(_text(legend_x + 5, y, str(index), 10, fill="#ffffff", weight="700"))
        elements.append(_text(legend_x + 26, y - 6, strategy, 12, weight="700"))
        elements.append(
            _text(
                legend_x + 26,
                y + 10,
                f"tokens {reduction:.1%} | WER {wer:.1%} | sim {speaker:.2f}",
                11,
                fill="#53606b",
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_listening_examples(path: Path, asr_rows: list[dict]) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in asr_rows:
        grouped.setdefault(str(row.get("clip_id", "")), []).append(row)

    selected_clip = _select_example_clip(grouped)
    rows = sorted(
        grouped.get(selected_clip, []),
        key=lambda row: _strategy_rank(str(row.get("strategy", ""))),
    )
    lines = [
        "# Listening Examples",
        "",
        "These files are generated benchmark artifacts and are not committed to git.",
        "",
        f"Selected clip: `{selected_clip}`",
        "",
        "| Strategy | WER | CER | Sample | Hypothesis |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        sample_name = Path(str(row.get("sample_path", ""))).name
        sample_path = f"samples/{sample_name}" if sample_name else ""
        lines.append(
            "| "
            f"`{row.get('strategy', '')}` | "
            f"{_to_float(row.get('wer')):.2%} | "
            f"{_to_float(row.get('cer')):.2%} | "
            f"`{sample_path}` | "
            f"{_escape_markdown(str(row.get('hypothesis_text', '')))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_energy_strategy(summary: dict[str, dict]) -> str:
    candidates = {
        strategy: values
        for strategy, values in summary.items()
        if strategy.startswith("energy")
    }
    if not candidates:
        return ""
    return min(
        candidates,
        key=lambda strategy: (
            float(candidates[strategy].get("mean_wer", 1.0)),
            -float(candidates[strategy].get("mean_speaker_similarity", 0.0)),
        ),
    )


def _select_example_clip(grouped: dict[str, list[dict]]) -> str:
    best_clip = ""
    best_score = -1.0
    for clip_id, rows in grouped.items():
        by_strategy = {str(row.get("strategy")): row for row in rows}
        baseline = _to_float(by_strategy.get("baseline", {}).get("wer"))
        uniform = _to_float(by_strategy.get("uniform", {}).get("wer"))
        salience_scores = [
            _to_float(row.get("wer"))
            for strategy, row in by_strategy.items()
            if "salience" in strategy or strategy.startswith("energy")
        ]
        if not salience_scores:
            continue
        score = uniform - min(salience_scores) - baseline
        if score > best_score:
            best_clip = clip_id
            best_score = score
    return best_clip


def _strategy_rank(strategy: str) -> int:
    if strategy in PREFERRED_EXAMPLE_STRATEGIES:
        return PREFERRED_EXAMPLE_STRATEGIES.index(strategy)
    if strategy.startswith("energy"):
        return 4
    return len(PREFERRED_EXAMPLE_STRATEGIES)


def _read_csv_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: list[dict], key: str) -> float:
    values = [_to_float(row.get(key)) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _to_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str = "#9aa7b2",
    dash: str = "",
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="1"{dash_attr}/>'
    )


def _text(
    x: float,
    y: float,
    value: str,
    size: int,
    fill: str = "#172026",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(value)}</text>'
    )


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _ordered_chart_points(points: list[ChartPoint]) -> list[ChartPoint]:
    preferred = {
        "baseline": 0,
        "uniform": 1,
        "acoustic_salience": 2,
        "energy_salience": 3,
        "energy_tuned_e4_t1_o2": 4,
        "patch": 99,
    }
    return sorted(points, key=lambda item: (preferred.get(item[0], 50), item[0]))


def _spread_chart_points(
    points: list[ChartPoint],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[PlacedChartPoint]:
    offsets = [
        (0, 0),
        (-54, -54),
        (-54, 0),
        (-54, 54),
        (0, -54),
        (0, 54),
        (54, -54),
        (54, 0),
        (54, 54),
        (-108, -54),
        (-108, 54),
        (108, -54),
        (108, 54),
    ]
    placed: list[PlacedChartPoint] = []
    for point in points:
        _strategy, x, y, radius, _reduction, _wer, _speaker = point
        candidates = []
        for dx, dy in offsets:
            display_x = max(x_min + radius, min(x_max - radius, x + dx))
            display_y = max(y_min + radius, min(y_max - radius, y + dy))
            candidates.append((display_x, display_y, dx * dx + dy * dy))
        for display_x, display_y, _distance in sorted(candidates, key=lambda item: item[2]):
            if all(
                _distance_between(display_x, display_y, other_x, other_y)
                >= radius + other_point[3] + 8
                for other_point, other_x, other_y in placed
            ):
                placed.append((point, display_x, display_y))
                break
        else:
            display_x, display_y, _distance = min(
                candidates,
                key=lambda candidate: _overlap_score(candidate[0], candidate[1], radius, placed),
            )
            placed.append((point, display_x, display_y))
    return placed


def _strategy_color(strategy: str, palette: dict[str, str]) -> str:
    if strategy in palette:
        return palette[strategy]
    if strategy.startswith("energy_tuned"):
        return "#8a7a25"
    return "#4d5f6f"


def _distance_between(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _overlap_score(
    x: float,
    y: float,
    radius: float,
    placed: list[PlacedChartPoint],
) -> float:
    score = 0.0
    for other_point, other_x, other_y in placed:
        clearance = _distance_between(x, y, other_x, other_y) - radius - other_point[3] - 8
        if clearance < 0:
            score += abs(clearance)
    return score
