from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audiotokenlab.models import ProfileConfig


def load_config(path: str | Path) -> ProfileConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)

    output_dir = Path(raw.get("output_dir", "runs/demo"))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()

    return ProfileConfig(
        run_id=str(raw.get("run_id", config_path.stem)),
        output_dir=output_dir,
        dataset=dict(raw.get("dataset", {})),
        tokenizer=dict(raw.get("tokenizer", {"name": "dummy"})),
        strategies=list(raw.get("strategies", [{"name": "baseline"}])),
        kv_cache=dict(raw.get("kv_cache", {})),
    )

