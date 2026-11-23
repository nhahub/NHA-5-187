"""FastAPI backend tests (M4-T02): one scan, several visits, error handling, and that uploads are not kept.

A stand-in segmenter replaces the exported model: it returns the T1 intensities as labels, in SimpleITK order
(z, y, x) like FastSegmenter, so the test can check that the saved mask ends up in the scan's own orientation.
"""
import importlib.util
import io
import json
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODALITIES = ("t1", "t1ce", "t2", "flair")


class FakeSegmenter:
    def segment(self, image_files):
        data = np.asarray(nib.load(image_files[0]).dataobj)
        return np.transpose(data, (2, 1, 0)).astype(np.uint8), {}, {"total_s": 0.01}


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINMRI_API_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setenv("BRAINMRI_EXPORT_DIR", str(tmp_path / "export"))
    monkeypatch.setenv("BRAINMRI_DEVICE", "cpu")
    spec = importlib.util.spec_from_file_location("brainmri_api_main", REPO / "api" / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_segmenter", lambda: FakeSegmenter())
    return module, TestClient(module.app), tmp_path / "outputs"


def nifti_bytes(labels):
    """A 1 mm NIfTI whose intensities are the given labels."""
    buffer = io.BytesIO()
    img = nib.Nifti1Image(labels.astype(np.float32), np.eye(4))
    file_map = nib.Nifti1Image.make_file_map({"image": buffer, "header": buffer})
    img.to_file_map(file_map)
    return buffer.getvalue()


def scan_files(labels, prefix=""):
    data = nifti_bytes(labels)
    return [(m, (f"{prefix}{m}.nii", data, "application/octet-stream")) for m in MODALITIES]


def tumor(shape, wt_voxels):
    """Labels with an asymmetric block: edema (2) with a small enhancing part (3); WT = wt_voxels."""
    labels = np.zeros(shape, np.uint8)
    flat = labels.reshape(-1)
    flat[:wt_voxels] = 2
    flat[:max(1, wt_voxels // 10)] = 3
    return labels


def test_predict_returns_volumes_and_keeps_the_scan_orientation(api):
    module, client, outputs = api
    labels = tumor((20, 16, 12), 600)
    response = client.post("/predict", files=scan_files(labels), data={"case_id": "c1", "run_radiomics": "false"})
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    volumes = result["segmentation"]["volumes"]
    assert volumes["WT"]["voxels"] == 600 and volumes["WT"]["volume_cm3"] == pytest.approx(0.6)
    assert volumes["ET"]["voxels"] == 60 and volumes["TC"]["voxels"] == 60
    mask = np.asarray(nib.load(result["segmentation"]["mask_path"]).dataobj)
    assert np.array_equal(mask, labels)                               # same voxels, same axis order as the input
    assert not list(outputs.rglob("inputs")), "uploaded scans must be deleted after the request"


def test_cube_shaped_scans_are_reoriented_too(api):
    module, client, _ = api
    labels = np.zeros((10, 10, 10), np.uint8)
    labels[0:2, 5:9, 7] = 3                                           # not symmetric under axis reversal
    response = client.post("/predict", files=scan_files(labels), data={"run_radiomics": "false"})
    assert response.status_code == 200, response.text
    mask = np.asarray(nib.load(response.json()["result"]["segmentation"]["mask_path"]).dataobj)
    assert np.array_equal(mask, labels)


def test_wrong_file_type_and_bad_geometry_are_refused(api):
    module, client, outputs = api
    files = scan_files(tumor((8, 8, 8), 10))
    files[0] = ("t1", ("t1.txt", b"not an image", "text/plain"))
    assert client.post("/predict", files=files).status_code == 415
    other = scan_files(tumor((9, 8, 8), 10))
    mixed = scan_files(tumor((8, 8, 8), 10))[:3] + other[3:]
    assert client.post("/predict", files=mixed, data={"run_radiomics": "false"}).status_code == 422
    assert not list(outputs.rglob("*.nii")), "nothing from a refused request is kept"


def test_longitudinal_percent_change_follows_the_1_cm3_rule(api):
    module, client, outputs = api
    shape = (20, 20, 10)                                              # 4000 voxels = 4 cm3
    visits = [("pre", 500), ("post", 1500), ("later", 3000)]          # WT 0.5, 1.5, 3.0 cm3
    files, meta = [], []
    for name, wt in visits:
        files += [("files", f) for _, f in scan_files(tumor(shape, wt), prefix=f"{name}_")]
        meta.append({"visit_id": name})
    response = client.post("/predict/longitudinal", files=files,
                           data={"patient_id": "p1", "visits_json": json.dumps(meta), "run_radiomics": "false"})
    assert response.status_code == 200, response.text
    changes = {(c["from_visit"], c["region"]): c for c in response.json()["longitudinal_changes"]}
    first, second = changes[("pre", "WT")], changes[("post", "WT")]
    assert first["change_cm3"] == pytest.approx(1.0) and first["change_percent"] is None
    assert "below 1 cm3" in first["note"]
    assert second["change_percent"] == pytest.approx(100.0) and second["note"] == ""
    assert not list(outputs.rglob("inputs"))


def test_longitudinal_refuses_repeated_visit_names(api):
    module, client, _ = api
    files = [("files", f) for _, f in scan_files(tumor((8, 8, 8), 10))] * 2
    meta = json.dumps([{"visit_id": "a"}, {"visit_id": "a"}])
    response = client.post("/predict/longitudinal", files=files,
                           data={"patient_id": "p", "visits_json": meta, "run_radiomics": "false"})
    assert response.status_code == 422


def test_old_results_are_purged(api):
    module, client, outputs = api
    old = outputs / "requests" / "old_case"
    old.mkdir(parents=True)
    (old / "mask.nii.gz").write_bytes(b"x")
    stale = time.time() - (module.KEEP_MINUTES + 5) * 60
    os.utime(old, (stale, stale))
    response = client.post("/predict", files=scan_files(tumor((8, 8, 8), 10)), data={"run_radiomics": "false"})
    assert response.status_code == 200, response.text
    assert not old.exists()
    assert client.get("/health").json()["results_kept_minutes"] == module.KEEP_MINUTES
