from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import modal


APP_DIR = Path("/root/audiotokenlab")
DEFAULT_CONFIG = "experiments/configs/encodec_demo.json"
DEFAULT_LOCAL_OUT = Path("modal-runs")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install("torch>=2.0", "encodec>=0.1.1")
    .add_local_dir("src", str(APP_DIR / "src"))
    .add_local_dir("experiments", str(APP_DIR / "experiments"))
    .env({"PYTHONPATH": str(APP_DIR / "src")})
    .workdir(str(APP_DIR))
)

app = modal.App("audiotokenlab", image=image)


@app.function(gpu="L4", timeout=20 * 60, memory=8192)
def run_encodec_profile(config_path: str = DEFAULT_CONFIG) -> dict:
    from audiotokenlab.runner import run_profile

    rows = run_profile(config_path)
    config = _load_json_config(APP_DIR / config_path)
    output_dir = _resolve_output_dir(APP_DIR / config_path, config.get("output_dir"))
    archive = _zip_directory(output_dir)
    return {
        "run_id": config.get("run_id", "encodec_demo"),
        "row_count": len(rows),
        "output_dir": str(output_dir),
        "archive_name": f"{config.get('run_id', 'encodec_demo')}.zip",
        "archive_bytes": archive,
    }


@app.local_entrypoint()
def main(
    config_path: str = DEFAULT_CONFIG,
    local_out: str = str(DEFAULT_LOCAL_OUT),
    extract: bool = True,
) -> None:
    result = run_encodec_profile.remote(config_path)
    target_root = Path(local_out)
    target_root.mkdir(parents=True, exist_ok=True)
    archive_path = target_root / str(result["archive_name"])
    archive_path.write_bytes(result["archive_bytes"])
    print(f"wrote archive: {archive_path}")
    print(f"remote rows: {result['row_count']}")

    if extract:
        extract_dir = target_root / str(result["run_id"])
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(result["archive_bytes"])) as archive:
            archive.extractall(extract_dir)
        print(f"extracted artifacts: {extract_dir}")


def _load_json_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_output_dir(config_path: Path, configured_output_dir: str | None) -> Path:
    output_dir = Path(configured_output_dir or "runs/encodec_demo")
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    return output_dir


def _zip_directory(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory))
    return buffer.getvalue()

