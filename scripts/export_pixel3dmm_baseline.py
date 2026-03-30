import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sorted_files(directory: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes],
        key=lambda path: path.name,
    )


def _copy_files(files: list[Path], destination_dir: Path) -> list[str]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied_names: list[str] = []
    for source_path in files:
        shutil.copy2(source_path, destination_dir / source_path.name)
        copied_names.append(source_path.name)
    return copied_names


def _infer_tracking_dir(tracking_root: Path, tracking_run_name: str | None) -> tuple[Path, str]:
    if tracking_run_name is not None:
        tracking_dir = tracking_root / tracking_run_name
        if not tracking_dir.is_dir():
            raise FileNotFoundError(f"Tracking run not found: {tracking_dir}")
        return tracking_dir, tracking_run_name

    candidates = sorted([path for path in tracking_root.iterdir() if path.is_dir()], key=lambda path: path.name)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one tracking run under {tracking_root}, found {[path.name for path in candidates]}"
        )
    return candidates[0], candidates[0].name


def export_baseline(
    method_name: str,
    sample_name: str,
    source_input_dir: Path,
    preprocessing_dir: Path,
    tracking_dir: Path,
    destination_root: Path,
    tracking_run_name: str,
) -> Path:
    cropped_dir = preprocessing_dir / "cropped"
    normals_dir = preprocessing_dir / "p3dmm" / "normals"
    uv_map_dir = preprocessing_dir / "p3dmm" / "uv_map"
    mica_dir = preprocessing_dir / "mica"
    mesh_dir = tracking_dir / "mesh"
    result_video = tracking_dir / "result.mp4"

    raw_inputs = _sorted_files(source_input_dir, (".jpg", ".jpeg", ".png"))
    cropped_frames = _sorted_files(cropped_dir, (".jpg", ".jpeg", ".png"))
    normal_predictions = _sorted_files(normals_dir, (".png",))
    uv_map_predictions = _sorted_files(uv_map_dir, (".png",))
    meshes = _sorted_files(mesh_dir, (".ply",))
    mica_runs = sorted([path.name for path in mica_dir.iterdir() if path.is_dir()]) if mica_dir.exists() else []

    if not cropped_frames:
        raise RuntimeError(f"No cropped frames found in {cropped_dir}")
    if len(normal_predictions) != len(cropped_frames):
        raise RuntimeError("Normals prediction count does not match cropped frame count")
    if len(uv_map_predictions) != len(cropped_frames):
        raise RuntimeError("UV-map prediction count does not match cropped frame count")
    if not meshes:
        raise RuntimeError(f"No meshes found in {mesh_dir}")
    if not result_video.is_file():
        raise FileNotFoundError(f"Missing result video: {result_video}")

    destination_dir = destination_root / method_name / sample_name
    copied_raw_inputs = _copy_files(raw_inputs, destination_dir / "inputs" / "raw")
    copied_cropped_frames = _copy_files(cropped_frames, destination_dir / "inputs" / "cropped")
    copied_normals = _copy_files(normal_predictions, destination_dir / "predictions" / "normals")
    copied_uv_maps = _copy_files(uv_map_predictions, destination_dir / "predictions" / "uv_map")
    copied_meshes = _copy_files(meshes, destination_dir / "reconstruction" / "meshes")
    _copy_files([result_video], destination_dir / "reconstruction" / "video")

    placeholder_frames = sorted(
        [frame_path.stem for frame_path in cropped_frames if frame_path.stem not in set(mica_runs)]
    )

    metadata = {
        "method": method_name,
        "sample": sample_name,
        "tracking_run_name": tracking_run_name,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "raw_inputs": str(source_input_dir.resolve()),
            "preprocessing": str(preprocessing_dir.resolve()),
            "tracking": str(tracking_dir.resolve()),
        },
        "frame_count": len(cropped_frames),
        "placeholder_frames": placeholder_frames,
        "files": {
            "raw_inputs": copied_raw_inputs,
            "cropped_frames": copied_cropped_frames,
            "normals": copied_normals,
            "uv_map": copied_uv_maps,
            "meshes": copied_meshes,
            "video": "result.mp4",
        },
    }
    metadata_path = destination_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Pixel3DMM outputs into a FaceForge baseline layout.")
    parser.add_argument("--sample-name", required=True, help="Sample name, for example 'tom'.")
    parser.add_argument("--method-name", default="pixel3dmm", help="Baseline method name.")
    parser.add_argument("--source-input-dir", type=Path, help="Raw input image directory.")
    parser.add_argument("--preprocessing-dir", type=Path, help="Pixel3DMM preprocessing output directory.")
    parser.add_argument("--tracking-root", type=Path, help="Pixel3DMM tracking root directory.")
    parser.add_argument("--tracking-run-name", help="Specific tracking run directory name.")
    parser.add_argument("--destination-root", type=Path, help="Baseline output root directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_input_dir = args.source_input_dir or (REPO_ROOT / "assets" / args.sample_name)
    preprocessing_dir = args.preprocessing_dir or (
        REPO_ROOT / "outputs" / "pixel3dmm_baseline" / "preprocessed" / args.sample_name
    )
    tracking_root = args.tracking_root or (REPO_ROOT / "outputs" / "pixel3dmm_baseline" / "tracking")
    destination_root = args.destination_root or (REPO_ROOT / "outputs" / "baselines")

    tracking_dir, tracking_run_name = _infer_tracking_dir(tracking_root, args.tracking_run_name)
    exported_dir = export_baseline(
        method_name=args.method_name,
        sample_name=args.sample_name,
        source_input_dir=source_input_dir,
        preprocessing_dir=preprocessing_dir,
        tracking_dir=tracking_dir,
        destination_root=destination_root,
        tracking_run_name=tracking_run_name,
    )
    print(exported_dir)


if __name__ == "__main__":
    main()
