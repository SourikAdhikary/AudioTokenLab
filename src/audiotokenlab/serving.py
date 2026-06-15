from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any


DEFAULT_STACK_SPEC = {
    "name": "reference_audio_token_transformer",
    "layers": 24,
    "hidden_size": 1024,
    "attention_heads": 16,
    "bytes_per_element": 2,
    "decode_steps": 128,
}


def write_serving_stack_report(
    output_dir: Path,
    stack_spec: dict[str, Any] | None = None,
    run_torch_microbench: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    metrics_rows = _read_csv_dicts(output_dir / "metrics.csv")
    spec = dict(DEFAULT_STACK_SPEC)
    if stack_spec:
        spec.update(stack_spec)

    strategy_summary = _summarize_serving_by_strategy(metrics_rows, spec)
    report: dict[str, Any] = {
        "stack": spec,
        "row_count": len(metrics_rows),
        "strategy_summary": strategy_summary,
    }
    if run_torch_microbench and metrics_rows:
        report["torch_microbench"] = benchmark_torch_transformer_stack(
            metrics_rows,
            device=device,
            hidden_size=int(spec["hidden_size"]),
            attention_heads=int(spec["attention_heads"]),
        )

    (output_dir / "serving_stack_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_serving_markdown(output_dir / "serving_stack_report.md", report)
    return report


def benchmark_torch_transformer_stack(
    metrics_rows: list[dict[str, str]],
    device: str,
    hidden_size: int,
    attention_heads: int,
    warmup: int = 1,
    iterations: int = 3,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for the serving microbenchmark") from exc

    selected_lengths = _representative_lengths(metrics_rows)
    if not selected_lengths:
        return {"row_count": 0}

    torch_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    layer = torch.nn.TransformerEncoderLayer(
        d_model=hidden_size,
        nhead=attention_heads,
        dim_feedforward=hidden_size * 4,
        batch_first=True,
    ).to(torch_device)
    layer.eval()
    timings = []
    with torch.inference_mode():
        for strategy, token_count in selected_lengths:
            length = max(1, min(int(token_count), 4096))
            hidden = torch.randn(1, length, hidden_size, device=torch_device)
            for _ in range(warmup):
                _ = layer(hidden)
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                _ = layer(hidden)
            if torch_device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0 / iterations
            timings.append(
                {
                    "strategy": strategy,
                    "tokens": length,
                    "prefill_ms": elapsed_ms,
                    "device": str(torch_device),
                }
            )
    return {
        "row_count": len(timings),
        "hidden_size": hidden_size,
        "attention_heads": attention_heads,
        "timings": timings,
    }


def _summarize_serving_by_strategy(
    rows: list[dict[str, str]],
    spec: dict[str, Any],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("strategy", "")), []).append(row)

    summary: dict[str, dict[str, float]] = {}
    for strategy, strategy_rows in sorted(grouped.items()):
        original_tokens = [_to_float(row.get("original_tokens")) for row in strategy_rows]
        compressed_tokens = [_to_float(row.get("compressed_tokens")) for row in strategy_rows]
        kv_savings = [_to_float(row.get("estimated_kv_cache_savings_mb")) for row in strategy_rows]
        reduction = [_to_float(row.get("token_reduction_ratio")) for row in strategy_rows]
        mean_original = statistics.fmean(original_tokens) if original_tokens else 0.0
        mean_compressed = statistics.fmean(compressed_tokens) if compressed_tokens else 0.0
        summary[strategy] = {
            "row_count": float(len(strategy_rows)),
            "mean_original_tokens": mean_original,
            "mean_compressed_tokens": mean_compressed,
            "mean_token_reduction_ratio": statistics.fmean(reduction) if reduction else 0.0,
            "mean_kv_cache_savings_mb": statistics.fmean(kv_savings) if kv_savings else 0.0,
            "prefill_attention_work_ratio": _attention_work_ratio(mean_original, mean_compressed),
            "decode_kv_read_reduction_ratio": _decode_kv_read_reduction(mean_original, mean_compressed),
            "estimated_decode_kv_read_mb_saved": _decode_kv_read_mb_saved(mean_original, mean_compressed, spec),
        }
    return summary


def _write_serving_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Serving Stack Report",
        "",
        f"Stack: `{report['stack']['name']}`",
        "",
        "| Strategy | Mean Tokens | Token Reduction | KV Savings | Prefill Work Ratio | Decode KV Read Reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, row in report.get("strategy_summary", {}).items():
        lines.append(
            "| "
            f"`{strategy}` | "
            f"{row['mean_compressed_tokens']:.1f} | "
            f"{row['mean_token_reduction_ratio']:.2%} | "
            f"{row['mean_kv_cache_savings_mb']:.2f} MB | "
            f"{row['prefill_attention_work_ratio']:.3f}x | "
            f"{row['decode_kv_read_reduction_ratio']:.2%} |"
        )
    if "torch_microbench" in report:
        lines.extend(["", "## PyTorch Microbenchmark", ""])
        for row in report["torch_microbench"].get("timings", []):
            lines.append(
                f"- `{row['strategy']}`: {row['tokens']} tokens, "
                f"{row['prefill_ms']:.2f} ms prefill on {row['device']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _representative_lengths(rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("strategy", "")), []).append(
            int(_to_float(row.get("compressed_tokens")))
        )
    return [
        (strategy, int(statistics.median(lengths)))
        for strategy, lengths in sorted(grouped.items())
        if lengths
    ]


def _attention_work_ratio(original_tokens: float, compressed_tokens: float) -> float:
    if original_tokens <= 0.0:
        return 0.0
    return (compressed_tokens * compressed_tokens) / (original_tokens * original_tokens)


def _decode_kv_read_reduction(original_tokens: float, compressed_tokens: float) -> float:
    if original_tokens <= 0.0:
        return 0.0
    return max(0.0, (original_tokens - compressed_tokens) / original_tokens)


def _decode_kv_read_mb_saved(
    original_tokens: float,
    compressed_tokens: float,
    spec: dict[str, Any],
) -> float:
    token_delta = max(0.0, original_tokens - compressed_tokens)
    bytes_per_token = (
        int(spec["layers"])
        * int(spec["hidden_size"])
        * 2
        * int(spec["bytes_per_element"])
    )
    return token_delta * bytes_per_token * int(spec["decode_steps"]) / (1024.0 * 1024.0)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)
