from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

BASE = Path(os.getenv("BRAINMRI_BASE", "/content/drive/MyDrive/BrainMRI_Data"))
EXPORT_DIR = Path(os.getenv("BRAINMRI_EXPORT_DIR", str(BASE / "models/seg-model-v1-torchscript")))
OUTPUT_ROOT = Path(os.getenv("BRAINMRI_API_OUTPUT", str(BASE / "runs/API_Backend")))
DEVICE = os.getenv("BRAINMRI_DEVICE", "cuda" if __import__("torch").cuda.is_available() else "cpu")
RUN_RADIOMICS_DEFAULT = os.getenv("API_RUN_RADIOMICS", "true").lower() in {"1", "true", "yes"}
# TTA can multiply inference time substantially. It is disabled by default; set API_TTA=true to enable it.
TTA = os.getenv("API_TTA", "false").lower() in {"1", "true", "yes"}
MAX_UPLOAD_MB = int(os.getenv("API_MAX_UPLOAD_MB", "512"))
# Uploaded scans are deleted as soon as a request finishes. Results (mask, radiomics CSV) stay for this many
# minutes so the web interface can open the mask, then they are purged.
KEEP_MINUTES = float(os.getenv("API_KEEP_MINUTES", "60"))
MODALITIES = ("t1", "t1ce", "t2", "flair")
REGIONS = {"TC": (1, 3), "WT": (1, 2, 3), "ET": (3,)}
# Same rule as bmts.quantitative.longitudinal (M3-T02): no percentage when the earlier volume is this small
MIN_BASELINE_CM3 = 1.0

app = FastAPI(
    title="Brain MRI Segmentation API",
    version="1.0.1",
    description="Project-compatible API for registered seg-model v1. Research prototype only."
)

_segmenter = None

def safe_id(value: str | None) -> str:
    value = (value or "case").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return (value or "case")[:80]

def is_nii_name(filename: str | None) -> bool:
    name = (filename or "").lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")

def get_segmenter():
    global _segmenter
    if _segmenter is None:
        if not EXPORT_DIR.is_dir() or not (EXPORT_DIR / "deploy_manifest.json").is_file():
            raise RuntimeError(f"Registered TorchScript export is missing: {EXPORT_DIR}")
        from bmts.deploy.segmenter import FastSegmenter
        _segmenter = FastSegmenter.from_dir(EXPORT_DIR, device=DEVICE, tta=TTA)
    return _segmenter

async def save_upload(upload: UploadFile, dest: Path) -> None:
    if not is_nii_name(upload.filename):
        raise HTTPException(status_code=415, detail=f"Unsupported file type for {upload.filename!r}; upload .nii or .nii.gz.")
    total = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.")
                out.write(chunk)
    finally:
        await upload.close()
    if total == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Empty upload: {upload.filename!r}")

def validate_modalities(paths: dict[str, Path]) -> dict[str, Any]:
    images = {}
    try:
        for name in MODALITIES:
            p = paths[name]
            try:
                img = nib.load(str(p))
                if len(img.shape) != 3:
                    raise ValueError(f"{name} must be a 3D NIfTI image; got shape {img.shape}.")
                if not np.all(np.isfinite(img.affine)):
                    raise ValueError(f"{name} has a non-finite affine matrix.")
                images[name] = img
            except Exception as exc:
                raise ValueError(f"Invalid or unreadable NIfTI for {name}: {exc}") from exc
        ref = images["t1"]
        for name, img in images.items():
            if img.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine, atol=1e-4):
                raise ValueError(f"Geometry mismatch: {name} shape/affine differs from t1.")
        voxel_volume_mm3 = float(abs(np.linalg.det(ref.affine[:3, :3])))
        return {
            "shape": [int(x) for x in ref.shape],
            "voxel_volume_mm3": voxel_volume_mm3,
            "voxel_spacing_mm": [float(x) for x in ref.header.get_zooms()[:3]],
            "geometry_validated": True,
        }
    finally:
        images.clear()

def normalize_segmentation_axes(seg: np.ndarray, reference_shape: tuple[int, ...]) -> np.ndarray:
    """Return the prediction in the reference NIfTI array order (i, j, k).

    FastSegmenter returns SimpleITK order (z, y, x), the reverse of nibabel's (i, j, k), so the axes are reversed
    whenever the shape allows it. This also covers scans whose three sizes are equal, where both orders have the
    same shape.
    """
    arr = np.asarray(seg)
    target = tuple(int(x) for x in reference_shape[:3])
    if arr.ndim == 3 and arr.shape == target[::-1]:
        return np.transpose(arr, (2, 1, 0))
    raise ValueError(f"Prediction shape {arr.shape} does not match the reversed input NIfTI shape {target[::-1]}.")


def percent_change(before_cm3: float, after_cm3: float) -> float | None:
    """(after - before) / before x 100, or None when the earlier volume is below MIN_BASELINE_CM3."""
    if before_cm3 < MIN_BASELINE_CM3:
        return None
    return round((after_cm3 - before_cm3) / before_cm3 * 100.0, 3)


def purge_old_runs() -> None:
    """Delete request folders older than KEEP_MINUTES (results of earlier requests)."""
    cutoff = time.time() - KEEP_MINUTES * 60
    for kind in ("requests", "longitudinal"):
        folder = OUTPUT_ROOT / kind
        if not folder.is_dir():
            continue
        for run_dir in folder.iterdir():
            try:
                if run_dir.is_dir() and run_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(run_dir, ignore_errors=True)
            except OSError:
                pass

def calculate_volumes(mask_path: Path, voxel_volume_mm3: float) -> dict[str, Any]:
    mask_img = nib.load(str(mask_path))
    mask = np.asarray(mask_img.dataobj)
    if mask.ndim != 3:
        raise ValueError("Predicted mask is not 3D.")
    result = {}
    for region, labels in REGIONS.items():
        voxels = int(np.isin(mask, labels).sum())
        result[region] = {
            "voxels": voxels,
            "volume_cm3": round(voxels * voxel_volume_mm3 / 1000.0, 6),
        }
    result["label_counts"] = {str(int(label)): int((mask == label).sum()) for label in np.unique(mask)}
    return result

def extract_radiomics(paths: dict[str, Path], mask_path: Path, out_csv: Path) -> dict[str, Any]:
    try:
        from radiomics import featureextractor
        import SimpleITK as sitk

        mask = sitk.ReadImage(str(mask_path))
        settings = {
            "binWidth": 25,
            "resampledPixelSpacing": None,
            "interpolator": "sitkBSpline",
            "normalize": False,
            "label": 1,
        }
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        rows = []
        for modality in ("t1ce", "t2", "flair"):
            image = sitk.ReadImage(str(paths[modality]))
            # Explicitly check dimensions and physical geometry; do not silently resize/transpose.
            if image.GetSize() != mask.GetSize():
                raise ValueError(
                    f"Radiomics geometry mismatch for {modality}: image size {image.GetSize()} "
                    f"vs mask size {mask.GetSize()}. Both files must be in the same NIfTI geometry "
                    "and must be read from the current request folder."
                )
            if (image.GetOrigin() != mask.GetOrigin() or image.GetSpacing() != mask.GetSpacing()
                    or image.GetDirection() != mask.GetDirection()):
                raise ValueError(f"Radiomics physical geometry mismatch for {modality} image and mask.")
            for label in (1, 2, 3):
                binary = sitk.Cast(mask == label, sitk.sitkUInt8)
                if int(sitk.GetArrayViewFromImage(binary).sum()) == 0:
                    continue
                features = extractor.execute(image, binary, label=1)
                row = {"modality": modality, "label": label}
                row.update({str(k): (float(v) if isinstance(v, (int, float, np.number)) else str(v))
                            for k, v in features.items() if not str(k).startswith("diagnostics_")})
                rows.append(row)
        if not rows:
            return {"status": "empty_mask", "csv_path": None, "rows": 0}
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return {"status": "success", "csv_path": str(out_csv), "rows": len(rows)}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "csv_path": None}

def run_one_case(case_id: str, paths: dict[str, Path], case_out: Path, run_radiomics: bool) -> dict[str, Any]:
    validation = validate_modalities(paths)
    case_out.mkdir(parents=True, exist_ok=True)
    mask_path = case_out / f"{case_id}_seg.nii.gz"
    try:
        segmenter = get_segmenter()
        # FastSegmenter expects file paths. Pass plain strings rather than Path objects
        # to avoid path-like tuple/type incompatibilities across repository versions.
        input_files = [str(paths[m]) for m in MODALITIES]
        segment_output = segmenter.segment(input_files)
        if not isinstance(segment_output, tuple) or len(segment_output) < 3:
            raise TypeError(
                "Unexpected FastSegmenter.segment() result. Expected (segmentation, output, timings); "
                f"received {type(segment_output).__name__}."
            )
        seg, _segment_output_path, timings = segment_output
        if isinstance(seg, (str, Path)):
            # Some versions return a saved mask path as the first result.
            seg = np.asarray(nib.load(str(seg)).dataobj)
        elif isinstance(seg, tuple):
            # Do not pass tuple objects to Path/nibabel; only accept a tuple that wraps one array.
            if len(seg) == 1:
                seg = seg[0]
            else:
                raise TypeError(
                    "FastSegmenter returned a tuple for the segmentation array; "
                    f"tuple length={len(seg)}. Check the installed segmenter API."
                )
        ref = nib.load(str(paths["t1"]))
        seg = normalize_segmentation_axes(np.asarray(seg), ref.shape)
        seg_img = nib.Nifti1Image(seg.astype(np.uint8), ref.affine, ref.header.copy())
        nib.save(seg_img, str(mask_path))
    except HTTPException:
        raise
    except Exception as exc:
        raise RuntimeError(f"Segmentation failed: {type(exc).__name__}: {exc}") from exc

    volumes = calculate_volumes(mask_path, validation["voxel_volume_mm3"])
    radiomics = (extract_radiomics(paths, mask_path, case_out / f"{case_id}_radiomics.csv")
                 if run_radiomics else {"status": "disabled", "csv_path": None, "rows": 0})
    return {
        "case_id": case_id,
        "input_validation": validation,
        "segmentation": {"status": "success", "mask_path": str(mask_path), "volumes": volumes},
        "inference_timings": {str(k): float(v) for k, v in timings.items()},
        "radiomics": radiomics,
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Brain MRI Segmentation API",
        "export_dir_exists": EXPORT_DIR.is_dir(),
        "export_manifest_exists": (EXPORT_DIR / "deploy_manifest.json").is_file(),
        "device": DEVICE,
        "tta_enabled": TTA,
        "results_kept_minutes": KEEP_MINUTES,
        "supported_file_extensions": [".nii", ".nii.gz"],
        "modalities_required": list(MODALITIES),
    }

@app.post("/predict")
async def predict(
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
    case_id: str = Form("case"),
    run_radiomics: bool = Form(RUN_RADIOMICS_DEFAULT),
):
    purge_old_runs()
    cid = safe_id(case_id)
    request_dir = OUTPUT_ROOT / "requests" / f"{cid}_{uuid.uuid4().hex[:8]}"
    inputs_dir = request_dir / "inputs"
    uploads = {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair}
    paths = {}
    try:
        for modality, upload in uploads.items():
            suffix = ".nii.gz" if (upload.filename or "").lower().endswith(".nii.gz") else ".nii"
            paths[modality] = inputs_dir / f"{cid}_{modality}{suffix}"
            await save_upload(upload, paths[modality])
        result = run_one_case(cid, paths, request_dir / "results", run_radiomics)
        return JSONResponse({"status": "success", "pipeline": "seg-model-v1-torchscript",
                             "result": result, "longitudinal_changes": None})
    except HTTPException:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise
    except ValueError as exc:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        shutil.rmtree(inputs_dir, ignore_errors=True)       # never keep the uploaded scans

@app.post("/predict/longitudinal")
async def predict_longitudinal(
    patient_id: str = Form(...),
    visits_json: str = Form(..., description='JSON array: [{"visit_id":"preop","t1":"filename.nii.gz","t1ce":"...","t2":"...","flair":"..."}]'),
    files: list[UploadFile] = File(..., description="Upload 4 files per visit, ordered by visits_json: t1, t1ce, t2, flair."),
    run_radiomics: bool = Form(RUN_RADIOMICS_DEFAULT),
):
    """Predict each visit for one patient. Files must be ordered visit-by-visit, 4 modalities per visit."""
    try:
        visits = json.loads(visits_json)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="visits_json must be valid JSON.") from exc
    if not isinstance(visits, list) or len(visits) < 2:
        raise HTTPException(status_code=422, detail="At least two visits are required.")
    if len(files) != 4 * len(visits):
        raise HTTPException(status_code=422, detail=f"Expected {4 * len(visits)} files (4 modalities per visit), received {len(files)}.")
    purge_old_runs()
    patient = safe_id(patient_id)
    request_dir = OUTPUT_ROOT / "longitudinal" / f"{patient}_{uuid.uuid4().hex[:8]}"
    outputs = []
    try:
        seen = set()
        for i, visit in enumerate(visits):
            if not isinstance(visit, dict):
                raise HTTPException(status_code=422, detail=f"Visit {i} must be a JSON object.")
            visit_id = safe_id(str(visit.get("visit_id", f"visit_{i+1}")))
            if visit_id in seen:
                raise HTTPException(status_code=422, detail=f"Visit id {visit_id!r} is used twice.")
            seen.add(visit_id)
            visit_dir = request_dir / visit_id
            input_dir = visit_dir / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            paths = {}
            for j, modality in enumerate(MODALITIES):
                upload = files[i * 4 + j]
                if not is_nii_name(upload.filename):
                    raise HTTPException(status_code=415, detail=f"Unsupported file type for {upload.filename!r}; upload .nii or .nii.gz.")
                suffix = ".nii.gz" if (upload.filename or "").lower().endswith(".nii.gz") else ".nii"
                path = input_dir / f"{visit_id}_{modality}{suffix}"
                await save_upload(upload, path)
                paths[modality] = path
            result = run_one_case(f"{patient}_{visit_id}", paths, visit_dir / "results", run_radiomics)
            shutil.rmtree(input_dir, ignore_errors=True)    # never keep the uploaded scans
            outputs.append({"visit_id": visit_id, **result})
        changes = []
        for before, after in zip(outputs, outputs[1:]):
            for region in ("TC", "WT", "ET"):
                v0 = before["segmentation"]["volumes"][region]["volume_cm3"]
                v1 = after["segmentation"]["volumes"][region]["volume_cm3"]
                pct = percent_change(v0, v1)
                changes.append({
                    "from_visit": before["visit_id"], "to_visit": after["visit_id"],
                    "region": region, "volume_before_cm3": v0, "volume_after_cm3": v1,
                    "change_cm3": round(v1 - v0, 6), "change_percent": pct,
                    "note": "" if pct is not None else
                            f"no percentage: earlier volume below {MIN_BASELINE_CM3:g} cm3",
                })
        return JSONResponse({"status": "success", "patient_id": patient,
                             "pipeline": "seg-model-v1-torchscript",
                             "visits": outputs, "longitudinal_changes": changes})
    except HTTPException:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise
    except ValueError as exc:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        for visit_dir in request_dir.glob("*/inputs"):
            shutil.rmtree(visit_dir, ignore_errors=True)

@app.get("/")
def root():
    return {"message": "Brain MRI API is running", "docs": "/docs", "health": "/health"}
