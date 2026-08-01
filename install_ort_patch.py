import site
import shutil
from pathlib import Path


def find_onnxruntime_dir() -> Path:
    for site_dir in site.getsitepackages():
        candidate = Path(site_dir) / "onnxruntime"
        if candidate.exists():
            return candidate

    raise RuntimeError("Could not find installed onnxruntime package")


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    ort_dir = find_onnxruntime_dir()

    src_patch = repo_dir / "yolo_patch_session_v3.py"
    dst_patch = ort_dir / "yolo_patch_session_v3.py"

    if not src_patch.exists():
        raise FileNotFoundError(f"Missing patch file: {src_patch}")

    shutil.copy(src_patch, dst_patch)
    print(f"Copied patch: {dst_patch}")

    init_path = ort_dir / "__init__.py"
    text = init_path.read_text()

    patch_text = """

# ---- YOLO patched InferenceSession ----
try:
    from .yolo_patch_session_v3 import PatchedInferenceSession as InferenceSession
except Exception as _ort_yolo_patch_error:
    print("Warning: failed to enable YOLO patched InferenceSession:", _ort_yolo_patch_error)
# ---- end YOLO patched InferenceSession ----
"""

    if "YOLO patched InferenceSession" not in text:
        init_path.write_text(text + patch_text)

    print(f"Patched init: {init_path}")


if __name__ == "__main__":
    main()
