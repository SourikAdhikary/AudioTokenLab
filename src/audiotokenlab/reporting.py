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
        "sample_dir": str(config.output_dir / "samples"),
        "summary": summarize(rows),
        "strategy_summary": summarize_by_strategy(rows),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def write_dashboard(path: Path, config: ProfileConfig, rows: list[MetricRow]) -> None:
    summary = summarize(rows)
    strategy_rows = "\n".join(
        _html_strategy_row(strategy, values)
        for strategy, values in summarize_by_strategy(rows).items()
    )
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
    {_summary_card("Mean SNR", f'{summary["mean_reconstruction_snr_db"]:.2f} dB')}
  </section>
  <h2>Strategy Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Strategy</th>
        <th>Rows</th>
        <th>Mean Reduction</th>
        <th>Mean KV Savings MB</th>
        <th>Mean MSE</th>
        <th>Mean SNR</th>
        <th>Mean RTF</th>
      </tr>
    </thead>
    <tbody>
      {strategy_rows}
    </tbody>
  </table>
  <h2>Clip Details</h2>
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
        <th>SNR dB</th>
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


def write_asr_dashboard(output_dir: Path, asr_rows: list[dict]) -> None:
    metrics_rows = _read_csv_dicts(output_dir / "metrics.csv")
    speaker_rows = _read_csv_dicts(output_dir / "speaker_metrics.csv")
    manifest = _read_json(output_dir / "manifest.json")
    run_id = str(manifest.get("run_id", output_dir.name))
    codec_summary = manifest.get("summary", {})
    codec_by_strategy = manifest.get("strategy_summary", {})
    asr_by_strategy = _summarize_asr_by_strategy(asr_rows)
    speaker_by_strategy = _summarize_speaker_by_strategy(speaker_rows)
    strategy_rows = "\n".join(
        _html_joint_strategy_row(
            strategy,
            codec_by_strategy.get(strategy, {}),
            asr,
            speaker_by_strategy.get(strategy, {}),
        )
        for strategy, asr in asr_by_strategy.items()
    )
    tradeoff_rows = "\n".join(
        _html_tradeoff_row(
            strategy,
            codec_by_strategy.get(strategy, {}),
            asr,
            speaker_by_strategy.get(strategy, {}),
        )
        for strategy, asr in asr_by_strategy.items()
    )
    failure_rows = "\n".join(
        _html_failure_row(row)
        for row in sorted(asr_rows, key=lambda item: float(item["wer"]), reverse=True)[:10]
    )
    metric_rows = "\n".join(_html_metric_dict_row(row) for row in metrics_rows)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AudioTokenLab ASR Report - {html.escape(run_id)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172026; }}
    main {{ max-width: 1280px; margin: 0 auto; }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    h2 {{ margin-top: 32px; }}
    p {{ color: #3f4c57; line-height: 1.45; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 24px 0; }}
    .metric {{ border: 1px solid #d7dee4; border-radius: 8px; padding: 12px; background: #f8fafb; }}
    .label {{ font-size: 12px; color: #53606b; text-transform: uppercase; letter-spacing: 0.04em; }}
    .value {{ font-size: 22px; font-weight: 650; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #d7dee4; padding: 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, .text {{ text-align: left; }}
    th {{ background: #eef3f6; }}
    audio {{ width: 180px; max-width: 100%; }}
    .note {{ border-left: 4px solid #8091a2; padding-left: 12px; }}
    .nowrap {{ white-space: nowrap; }}
  </style>
</head>
<body>
<main>
  <h1>AudioTokenLab ASR Benchmark</h1>
  <p class="note">Run <strong>{html.escape(run_id)}</strong>. This dashboard combines codec metrics with downstream ASR WER/CER. Lower WER at the same token budget is the main signal.</p>
  <section class="summary">
    {_summary_card("Codec Rows", codec_summary.get("row_count", len(metrics_rows)))}
    {_summary_card("ASR Rows", len(asr_rows))}
    {_summary_card("Clips", codec_summary.get("clip_count", "-"))}
    {_summary_card("Mean Token Reduction", _format_percent(codec_summary.get("mean_token_reduction_ratio")))}
    {_summary_card("Mean KV Savings", _format_mb(codec_summary.get("mean_kv_cache_savings_mb")))}
    {_summary_card("Mean ASR WER", _format_percent(_mean_float(asr_rows, "wer")))}
    {_summary_card("Mean Speaker Sim", _format_number(_mean_float(speaker_rows, "speaker_similarity"), digits=3))}
  </section>
  <h2>Strategy Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Strategy</th>
        <th>Rows</th>
        <th>Token Reduction</th>
        <th>KV Savings MB</th>
        <th>SNR dB</th>
        <th>WER</th>
        <th>WER 95% CI</th>
        <th>CER</th>
        <th>CER 95% CI</th>
        <th>Speaker Sim</th>
      </tr>
    </thead>
    <tbody>
      {strategy_rows}
    </tbody>
  </table>
  <h2>Token Budget vs ASR</h2>
  <table>
    <thead>
      <tr>
        <th>Strategy</th>
        <th>Token Reduction</th>
        <th>WER</th>
        <th>WER 95% CI</th>
        <th>CER</th>
        <th>Speaker Sim</th>
        <th>KV Savings MB</th>
        <th>RTF</th>
      </tr>
    </thead>
    <tbody>
      {tradeoff_rows}
    </tbody>
  </table>
  <h2>Worst ASR Cases</h2>
  <table>
    <thead>
      <tr>
        <th>Clip</th>
        <th>Strategy</th>
        <th>WER</th>
        <th>CER</th>
        <th class="text">Reference</th>
        <th class="text">Hypothesis</th>
        <th>Sample</th>
      </tr>
    </thead>
    <tbody>
      {failure_rows}
    </tbody>
  </table>
  <h2>Codec Rows</h2>
  <table>
    <thead>
      <tr>
        <th>Clip</th>
        <th>Strategy</th>
        <th>Original Tokens</th>
        <th>Compressed Tokens</th>
        <th>Reduction</th>
        <th>KV Savings MB</th>
        <th>SNR dB</th>
        <th>RTF</th>
      </tr>
    </thead>
    <tbody>
      {metric_rows}
    </tbody>
  </table>
</main>
</body>
</html>
"""
    (output_dir / "dashboard.html").write_text(document, encoding="utf-8")


def summarize(rows: list[MetricRow]) -> dict:
    if not rows:
        return {
            "row_count": 0,
            "clip_count": 0,
            "mean_token_reduction_ratio": 0.0,
            "mean_kv_cache_savings_mb": 0.0,
            "mean_reconstruction_mse": 0.0,
            "mean_reconstruction_snr_db": 0.0,
        }
    return {
        "row_count": len(rows),
        "clip_count": len({row.clip_id for row in rows}),
        "mean_token_reduction_ratio": sum(row.token_reduction_ratio for row in rows)
        / len(rows),
        "mean_kv_cache_savings_mb": sum(row.estimated_kv_cache_savings_mb for row in rows)
        / len(rows),
        "mean_reconstruction_mse": sum(row.reconstruction_mse for row in rows) / len(rows),
        "mean_reconstruction_snr_db": sum(row.reconstruction_snr_db for row in rows)
        / len(rows),
    }


def summarize_by_strategy(rows: list[MetricRow]) -> dict[str, dict]:
    grouped: dict[str, list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault(row.strategy, []).append(row)

    summary: dict[str, dict] = {}
    for strategy in sorted(grouped):
        strategy_rows = grouped[strategy]
        summary[strategy] = {
            "row_count": len(strategy_rows),
            "mean_token_reduction_ratio": sum(
                row.token_reduction_ratio for row in strategy_rows
            )
            / len(strategy_rows),
            "mean_kv_cache_savings_mb": sum(
                row.estimated_kv_cache_savings_mb for row in strategy_rows
            )
            / len(strategy_rows),
            "mean_reconstruction_mse": sum(
                row.reconstruction_mse for row in strategy_rows
            )
            / len(strategy_rows),
            "mean_reconstruction_snr_db": sum(
                row.reconstruction_snr_db for row in strategy_rows
            )
            / len(strategy_rows),
            "mean_real_time_factor": sum(row.real_time_factor for row in strategy_rows)
            / len(strategy_rows),
        }
    return summary


def _summary_card(label: str, value: object) -> str:
    return (
        '<div class="metric">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div>'
        "</div>"
    )


def _html_strategy_row(strategy: str, values: dict) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(strategy)}</td>"
        f"<td>{values['row_count']}</td>"
        f"<td>{values['mean_token_reduction_ratio']:.2%}</td>"
        f"<td>{values['mean_kv_cache_savings_mb']:.2f}</td>"
        f"<td>{values['mean_reconstruction_mse']:.5f}</td>"
        f"<td>{values['mean_reconstruction_snr_db']:.2f}</td>"
        f"<td>{values['mean_real_time_factor']:.4f}</td>"
        "</tr>"
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
        f"<td>{row.reconstruction_snr_db:.2f}</td>"
        f"<td>{row.real_time_factor:.4f}</td>"
        "</tr>"
    )


def _read_csv_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_asr_by_strategy(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["strategy"]), []).append(row)

    summary: dict[str, dict] = {}
    for strategy in sorted(grouped):
        strategy_rows = grouped[strategy]
        summary[strategy] = {
            "row_count": len(strategy_rows),
            "mean_wer": _mean_float(strategy_rows, "wer"),
            "wer_ci95": _bootstrap_mean_ci(strategy_rows, "wer"),
            "mean_cer": _mean_float(strategy_rows, "cer"),
            "cer_ci95": _bootstrap_mean_ci(strategy_rows, "cer"),
        }
    return summary


def _summarize_speaker_by_strategy(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["strategy"]), []).append(row)

    summary: dict[str, dict] = {}
    for strategy in sorted(grouped):
        strategy_rows = grouped[strategy]
        summary[strategy] = {
            "row_count": len(strategy_rows),
            "mean_speaker_similarity": _mean_float(
                strategy_rows,
                "speaker_similarity",
            ),
        }
    return summary


def _html_joint_strategy_row(
    strategy: str,
    codec: dict,
    asr: dict,
    speaker: dict,
) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(strategy)}</td>"
        f"<td>{asr.get('row_count', 0)}</td>"
        f"<td>{_format_percent(codec.get('mean_token_reduction_ratio'))}</td>"
        f"<td>{_format_number(codec.get('mean_kv_cache_savings_mb'))}</td>"
        f"<td>{_format_number(codec.get('mean_reconstruction_snr_db'))}</td>"
        f"<td>{_format_percent(asr.get('mean_wer'))}</td>"
        f"<td class=\"nowrap\">{_format_ci_percent(asr.get('wer_ci95'))}</td>"
        f"<td>{_format_percent(asr.get('mean_cer'))}</td>"
        f"<td class=\"nowrap\">{_format_ci_percent(asr.get('cer_ci95'))}</td>"
        f"<td>{_format_number(speaker.get('mean_speaker_similarity'), digits=3)}</td>"
        "</tr>"
    )


def _html_tradeoff_row(strategy: str, codec: dict, asr: dict, speaker: dict) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(strategy)}</td>"
        f"<td>{_format_percent(codec.get('mean_token_reduction_ratio'))}</td>"
        f"<td>{_format_percent(asr.get('mean_wer'))}</td>"
        f"<td class=\"nowrap\">{_format_ci_percent(asr.get('wer_ci95'))}</td>"
        f"<td>{_format_percent(asr.get('mean_cer'))}</td>"
        f"<td>{_format_number(speaker.get('mean_speaker_similarity'), digits=3)}</td>"
        f"<td>{_format_number(codec.get('mean_kv_cache_savings_mb'))}</td>"
        f"<td>{_format_number(codec.get('mean_real_time_factor'), digits=4)}</td>"
        "</tr>"
    )


def _html_failure_row(row: dict) -> str:
    sample_name = Path(str(row.get("sample_path", ""))).name
    sample_href = f"samples/{sample_name}" if sample_name else ""
    if sample_href:
        sample_cell = (
            f'<audio controls src="{html.escape(sample_href)}"></audio>'
        )
    else:
        sample_cell = ""
    return (
        "<tr>"
        f"<td>{html.escape(str(row.get('clip_id', '')))}</td>"
        f"<td>{html.escape(str(row.get('strategy', '')))}</td>"
        f"<td>{_format_percent(row.get('wer'))}</td>"
        f"<td>{_format_percent(row.get('cer'))}</td>"
        f"<td class=\"text\">{html.escape(str(row.get('reference_text', '')))}</td>"
        f"<td class=\"text\">{html.escape(str(row.get('hypothesis_text', '')))}</td>"
        f"<td>{sample_cell}</td>"
        "</tr>"
    )


def _html_metric_dict_row(row: dict) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(str(row.get('clip_id', '')))}</td>"
        f"<td>{html.escape(str(row.get('strategy', '')))}</td>"
        f"<td>{html.escape(str(row.get('original_tokens', '')))}</td>"
        f"<td>{html.escape(str(row.get('compressed_tokens', '')))}</td>"
        f"<td>{_format_percent(row.get('token_reduction_ratio'))}</td>"
        f"<td>{_format_number(row.get('estimated_kv_cache_savings_mb'))}</td>"
        f"<td>{_format_number(row.get('reconstruction_snr_db'))}</td>"
        f"<td>{_format_number(row.get('real_time_factor'), digits=4)}</td>"
        "</tr>"
    )


def _mean_float(rows: list[dict], key: str) -> float:
    values = [_to_float(row.get(key)) for row in rows if row.get(key) is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _bootstrap_mean_ci(
    rows: list[dict],
    key: str,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1337,
) -> dict[str, float]:
    import random

    values = [_to_float(row.get(key)) for row in rows if row.get(key) is not None]
    if not values:
        return {"low": 0.0, "high": 0.0}
    if len(values) == 1:
        return {"low": values[0], "high": values[0]}

    rng = random.Random(seed + sum(ord(char) for char in key) + len(values))
    means: list[float] = []
    count = len(values)
    for _ in range(iterations):
        total = 0.0
        for _sample in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)

    means.sort()
    alpha = 1.0 - confidence
    low_index = max(0, min(len(means) - 1, int((alpha / 2.0) * len(means))))
    high_index = max(0, min(len(means) - 1, int((1.0 - alpha / 2.0) * len(means)) - 1))
    return {"low": means[low_index], "high": means[high_index]}


def _format_percent(value: object) -> str:
    return f"{_to_float(value):.2%}"


def _format_ci_percent(value: object) -> str:
    if not isinstance(value, dict):
        return "0.00%-0.00%"
    return f"{_to_float(value.get('low')):.2%}-{_to_float(value.get('high')):.2%}"


def _format_mb(value: object) -> str:
    return f"{_to_float(value):.2f} MB"


def _format_number(value: object, digits: int = 2) -> str:
    return f"{_to_float(value):.{digits}f}"


def _to_float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)
