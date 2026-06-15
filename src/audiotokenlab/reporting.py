from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from audiotokenlab.models import MetricRow, ProfileConfig


def write_run_artifacts(config: ProfileConfig, rows: list[MetricRow]) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(config.output_dir / "metrics.csv", rows)
    write_manifest(config.output_dir / "manifest.json", config, rows)
    write_dashboard(config.output_dir / "dashboard.html", config, rows)


def write_metrics_csv(path: Path, rows: list[MetricRow]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].to_dict().keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def write_manifest(path: Path, config: ProfileConfig, rows: list[MetricRow]) -> None:
    manifest = {
        "run_id": config.run_id,
        "row_count": len(rows),
        "strategies": sorted({row.strategy for row in rows}),
        "clip_count": len({row.clip_id for row in rows}),
        "output_dir": str(config.output_dir),
        "summary": summarize(rows),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def write_dashboard(path: Path, config: ProfileConfig, rows: list[MetricRow]) -> None:
    summary = summarize(rows)
    table_rows = "\n".join(_html_row(row) for row in rows)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AudioTokenLab Report - {html.escape(config.run_id)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172026; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 24px 0; }}
    .metric {{ border: 1px solid #d7dee4; border-radius: 8px; padding: 12px; background: #f8fafb; }}
    .label {{ font-size: 12px; color: #53606b; text-transform: uppercase; letter-spacing: 0.04em; }}
    .value {{ font-size: 22px; font-weight: 650; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #d7dee4; padding: 8px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #eef3f6; }}
  </style>
</head>
<body>
<main>
  <h1>AudioTokenLab Report</h1>
  <p>Run <strong>{html.escape(config.run_id)}</strong>. Metrics are generated from the current profiling pipeline.</p>
  <section class="summary">
    {_summary_card("Rows", summary["row_count"])}
    {_summary_card("Clips", summary["clip_count"])}
    {_summary_card("Mean Token Reduction", f'{summary["mean_token_reduction_ratio"]:.2%}')}
    {_summary_card("Mean KV Savings", f'{summary["mean_kv_cache_savings_mb"]:.2f} MB')}
    {_summary_card("Mean MSE", f'{summary["mean_reconstruction_mse"]:.5f}')}
  </section>
  <table>
    <thead>
      <tr>
        <th>Clip</th>
        <th>Strategy</th>
        <th>Original Tokens</th>
        <th>Compressed Tokens</th>
        <th>Reduction</th>
        <th>KV Cache MB</th>
        <th>KV Savings MB</th>
        <th>MSE</th>
        <th>RTF</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def summarize(rows: list[MetricRow]) -> dict:
    if not rows:
        return {
            "row_count": 0,
            "clip_count": 0,
            "mean_token_reduction_ratio": 0.0,
            "mean_kv_cache_savings_mb": 0.0,
            "mean_reconstruction_mse": 0.0,
        }
    return {
        "row_count": len(rows),
        "clip_count": len({row.clip_id for row in rows}),
        "mean_token_reduction_ratio": sum(row.token_reduction_ratio for row in rows)
        / len(rows),
        "mean_kv_cache_savings_mb": sum(row.estimated_kv_cache_savings_mb for row in rows)
        / len(rows),
        "mean_reconstruction_mse": sum(row.reconstruction_mse for row in rows) / len(rows),
    }


def _summary_card(label: str, value: object) -> str:
    return (
        '<div class="metric">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div>'
        "</div>"
    )


def _html_row(row: MetricRow) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(row.clip_id)}</td>"
        f"<td>{html.escape(row.strategy)}</td>"
        f"<td>{row.original_tokens}</td>"
        f"<td>{row.compressed_tokens}</td>"
        f"<td>{row.token_reduction_ratio:.2%}</td>"
        f"<td>{row.estimated_kv_cache_mb:.2f}</td>"
        f"<td>{row.estimated_kv_cache_savings_mb:.2f}</td>"
        f"<td>{row.reconstruction_mse:.5f}</td>"
        f"<td>{row.real_time_factor:.4f}</td>"
        "</tr>"
    )

