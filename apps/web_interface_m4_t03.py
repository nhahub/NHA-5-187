"""M4-T03 Brain MRI web interface.

Runs with:
    streamlit run apps/web_interface.py

The app calls the existing FastAPI /predict endpoint and reuses the
M2-T05 viewer helpers to display the returned segmentation mask.
"""
from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
)

from apps.m2_t05_viewer import load_volume, render_slice

MODALITIES = ("t1", "t1ce", "t2", "flair")
API_DEFAULT = os.getenv("BRAINMRI_API_URL", "http://127.0.0.1:8000")


def load_uploaded_nifti(uploaded_file) -> object:
    """Load an uploaded NIfTI file using the existing viewer's canonical loader."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif filename.endswith(".nii"):
        suffix = ".nii"
    else:
        raise ValueError(f"Unsupported NIfTI extension: {uploaded_file.name}")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        return load_volume(temp_path)
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def get_measurement_rows(volumes: dict) -> list[dict]:
    rows = []
    region_names = {
        "TC": "Tumor Core (TC)",
        "WT": "Whole Tumor (WT)",
        "ET": "Enhancing Tumor (ET)",
    }
    for key, label in region_names.items():
        values = volumes.get(key)
        if isinstance(values, dict):
            rows.append(
                {
                    "Region": label,
                    "Voxels": values.get("voxels"),
                    "Volume (cm³)": values.get("volume_cm3"),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Sidebar: API connection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Backend Connection")
    api_base_url = st.text_input(
        "FastAPI Base URL",
        value=API_DEFAULT,
        help="Use http://127.0.0.1:8000 when this Streamlit app runs in the same Colab runtime as FastAPI.",
    ).strip().rstrip("/")

    if st.button("Test Backend Connection", use_container_width=True):
        if not api_base_url:
            st.error("Please enter the API base URL.")
        else:
            try:
                health_response = requests.get(
                    f"{api_base_url}/health",
                    timeout=15,
                )
                health_response.raise_for_status()
                health_data = health_response.json()
                if health_data.get("status") == "ok":
                    st.success("Backend is connected!")
                    st.json(health_data)
                else:
                    st.warning("The API responded, but its status is not OK.")
                    st.json(health_data)
            except requests.RequestException as exc:
                st.error(f"Could not connect to the API: {exc}")


# ---------------------------------------------------------------------------
# Main page: upload and inference
# ---------------------------------------------------------------------------
st.title("🧠 Brain MRI Analysis")
st.caption("Upload four MRI modalities, run segmentation, and inspect the result.")
st.warning("Research prototype only — not intended for clinical diagnosis.")

st.subheader("Step 1 — Upload MRI files")
st.write(
    "Upload one NIfTI file for each modality. Supported formats: `.nii` and `.nii.gz`. "
    "All four scans should belong to the same case and share the same voxel geometry."
)

with st.form("mri_analysis_form"):
    st.caption("A unique case reference will be generated automatically when you start the analysis.")

    col1, col2 = st.columns(2)
    with col1:
        t1_file = st.file_uploader(
            "T1",
            type=["nii", "gz"],
            key="upload_t1",
        )
        t1ce_file = st.file_uploader(
            "T1ce",
            type=["nii", "gz"],
            key="upload_t1ce",
        )
    with col2:
        t2_file = st.file_uploader(
            "T2",
            type=["nii", "gz"],
            key="upload_t2",
        )
        flair_file = st.file_uploader(
            "FLAIR",
            type=["nii", "gz"],
            key="upload_flair",
        )

    submitted = st.form_submit_button(
        "Analyze Scan",
        type="primary",
        use_container_width=True,
    )

if submitted:
    uploads = {
        "t1": t1_file,
        "t1ce": t1ce_file,
        "t2": t2_file,
        "flair": flair_file,
    }

    missing = [name.upper() for name, item in uploads.items() if item is None]
    invalid = [
        name.upper()
        for name, item in uploads.items()
        if item is not None
        and not item.name.lower().endswith((".nii", ".nii.gz"))
    ]

    if missing:
        st.error("Missing files: " + ", ".join(missing))
    elif invalid:
        st.error("Unsupported file format for: " + ", ".join(invalid))
    elif not api_base_url:
        st.error("Please enter the FastAPI Base URL in the sidebar.")
    else:
        # Generate one internal ID per submitted analysis; users do not enter it.
        case_id = f"case_{uuid.uuid4().hex[:12]}"

        files_payload = {
            name: (
                uploaded.name,
                uploaded.getvalue(),
                "application/octet-stream",
            )
            for name, uploaded in uploads.items()
        }

        try:
            with st.spinner(
                "Running brain MRI segmentation. This may take a few minutes on CPU..."
            ):
                request_started = time.time()
                api_response = requests.post(
                    f"{api_base_url}/predict",
                    files=files_payload,
                    data={
                        "case_id": case_id,
                        "run_radiomics": "false",
                    },
                    timeout=(30, 1800),
                )
                request_elapsed = time.time() - request_started

            if not api_response.ok:
                st.error(
                    f"Segmentation request failed (HTTP {api_response.status_code})."
                )
                st.code(api_response.text[:5000])
            else:
                api_data = api_response.json()
                result = api_data.get("result", {})
                segmentation = result.get("segmentation", {})

                if segmentation.get("status") != "success":
                    st.error("The API responded, but segmentation was not successful.")
                    st.json(api_data)
                else:
                    mask_path_value = segmentation.get("mask_path")
                    if not mask_path_value:
                        st.error("The API response did not include segmentation.mask_path.")
                        st.json(api_data)
                    else:
                        mask_path = Path(mask_path_value)
                        if not mask_path.is_absolute():
                            mask_path = Path.cwd() / mask_path

                        if not mask_path.is_file():
                            st.error(
                                "The API returned a mask path that is not accessible "
                                "from this Streamlit process."
                            )
                            st.code(str(mask_path))
                        else:
                            with st.spinner("Loading MRI volumes and segmentation mask..."):
                                images = {
                                    name: load_uploaded_nifti(uploaded)
                                    for name, uploaded in uploads.items()
                                }
                                mask = load_volume(str(mask_path))

                            reference_shape = images["t1"].shape
                            mismatched = {
                                name: image.shape
                                for name, image in images.items()
                                if image.shape != reference_shape
                            }
                            if mismatched or mask.shape != reference_shape:
                                st.error("MRI and segmentation dimensions do not match.")
                                st.write("T1 shape:", reference_shape)
                                st.write("Other MRI shapes:", mismatched)
                                st.write("Mask shape:", mask.shape)
                            else:
                                st.session_state["mri_analysis"] = {
                                    "case_id": case_id,
                                    "images": images,
                                    "mask": mask,
                                    "api_data": api_data,
                                    "request_elapsed": request_elapsed,
                                    "mask_path": str(mask_path),
                                }
                                st.success(
                                    f"Segmentation completed successfully. Reference: {case_id}"
                                )

        except requests.Timeout:
            st.error(
                "The prediction request timed out. Check the backend logs before retrying; "
                "the model may still have been processing the scan."
            )
        except requests.RequestException as exc:
            st.error(f"Could not send the prediction request: {exc}")
        except (ValueError, KeyError, OSError) as exc:
            st.error(f"Could not load the API result or NIfTI files: {exc}")


# ---------------------------------------------------------------------------
# Results: measurements and interactive MRI overlay viewer
# ---------------------------------------------------------------------------
analysis = st.session_state.get("mri_analysis")

if analysis:
    st.divider()
    st.subheader("Step 2 — Segmentation results")

    api_data = analysis["api_data"]
    result = api_data.get("result", {})
    segmentation = result.get("segmentation", {})
    volumes = segmentation.get("volumes", {})
    timings = result.get("inference_timings", {})

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Case ID", analysis["case_id"])
    metric_col2.metric(
        "Request time",
        f'{analysis["request_elapsed"]:.1f} s',
    )
    if timings.get("total_s") is not None:
        metric_col3.metric("Inference time", f'{timings["total_s"]:.1f} s')
    else:
        metric_col3.metric("Segmentation", "Success")

    measurement_rows = get_measurement_rows(volumes)
    if measurement_rows:
        st.markdown("#### Tumor measurements")
        st.dataframe(
            pd.DataFrame(measurement_rows),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No TC/WT/ET volume measurements were returned by the API.")

    st.subheader("Step 3 — MRI viewer with segmentation overlay")
    images = analysis["images"]
    mask = analysis["mask"]
    volume_shape = images["t1"].shape

    control1, control2, control3, control4 = st.columns([1, 1.5, 1, 1])
    with control1:
        selected_modality = st.selectbox(
            "MRI modality",
            options=MODALITIES,
            index=3,
            format_func=lambda value: value.upper(),
        )
    with control2:
        slice_index = st.slider(
            "Axial slice",
            min_value=0,
            max_value=volume_shape[2] - 1,
            value=volume_shape[2] // 2,
            step=1,
        )
    with control3:
        opacity = st.slider(
            "Mask opacity",
            min_value=0.10,
            max_value=0.80,
            value=0.45,
            step=0.05,
        )
    with control4:
        show_mask = st.checkbox("Show tumor mask", value=True)

    fig = render_slice(
        images[selected_modality],
        mask,
        slice_index,
        opacity=opacity,
        show_mask=show_mask,
    )
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)

    st.caption(
        "Mask labels present: "
        + ", ".join(str(int(value)) for value in np.unique(mask) if int(value) != 0)
    )

    with st.expander("Inference details and output files"):
        st.write("Segmentation status:", segmentation.get("status"))
        st.write("Mask path:", analysis["mask_path"])
        st.write("Mask shape:", mask.shape)
        st.write("Unique mask labels:", [int(x) for x in np.unique(mask)])
        if timings:
            st.markdown("**Inference timings**")
            st.json(timings)
        if result.get("input_validation"):
            st.markdown("**Input validation**")
            st.json(result["input_validation"])
else:
    st.info(
        "Upload the four MRI files and click Analyze Scan. "
        "The segmentation results and viewer will appear here after a successful prediction."
    )
