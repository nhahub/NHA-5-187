"""M4-T03 Brain MRI web interface, written for people without medical training.

Runs with:
    streamlit run apps/web_interface_m4_t03.py

The app calls the FastAPI backend (/predict for one scan, /predict/longitudinal for several scans of the same person)
and reuses the M2-T05 viewer helpers to show the returned tumor outline on the scan. Every MRI type and tumor region
is explained in plain words; technical details stay in collapsed sections.
"""
from __future__ import annotations

import json
import os
import re
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
    page_title="Brain MRI Tumor Measurement",
    page_icon="🧠",
    layout="wide",
)

from apps.m2_t05_viewer import LABEL_COLORS, load_volume, render_slice

MODALITIES = ("t1", "t1ce", "t2", "flair")
API_DEFAULT = os.getenv("BRAINMRI_API_URL", "http://127.0.0.1:8000")
MAX_VISITS = 4

# Each MRI type: (short name, what it shows in plain words)
MODALITY_INFO = {
    "t1": ("T1", "the standard picture of the brain"),
    "t1ce": ("T1ce (T1 with contrast)", "taken after a contrast dye injection; the active part of a tumor lights up"),
    "t2": ("T2", "shows fluid and swelling"),
    "flair": ("FLAIR", "shows swelling around a tumor, with the normal brain fluid made dark"),
}
# Each region the model measures: (plain name, explanation)
REGION_INFO = {
    "WT": ("Whole abnormal area", "the tumor together with the swelling around it"),
    "TC": ("Tumor core", "the tumor itself, without the swelling"),
    "ET": ("Active part", "the part of the core that lights up after the contrast dye"),
}
# Legend for the viewer's label colors (the colors come from apps/m2_t05_viewer.LABEL_COLORS)
LABEL_LEGEND = [
    (LABEL_COLORS[1], "Inner part of the tumor"),
    (LABEL_COLORS[2], "Swelling around the tumor"),
    (LABEL_COLORS[3], "Active part of the tumor"),
    (LABEL_COLORS[4], "Space left by earlier surgery"),
]
# Words in a file name that only say which kind of image it is (removed when comparing file names)
MODALITY_WORDS = {"t1", "t1ce", "t1c", "t1gd", "gd", "t2", "flair", "contrast", "t2flair", "nii", "gz"}
SAME_CM3 = 0.1          # changes smaller than this are described as "about the same"

DISCLAIMER = (
    "This is a research tool for doctors. It measures what a computer model outlines on the scan. "
    "It cannot say what kind of tumor it is, how serious it is, or which treatment is needed. "
    "Always discuss the results with your doctor."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_uploaded_nifti(uploaded_file) -> np.ndarray:
    """Load an uploaded NIfTI file with the viewer's loader (the file is only kept in a temporary file)."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif filename.endswith(".nii"):
        suffix = ".nii"
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        return load_volume(temp_path)
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def guess_modality(filename: str) -> str | None:
    """MRI type from a file name such as BraTS2021_00110_t1ce.nii.gz; None when it cannot be told."""
    stem = filename.lower().replace(".nii.gz", "").replace(".nii", "")
    if "flair" in stem:
        return "flair"
    if any(tag in stem for tag in ("t1ce", "t1c", "t1gd", "t1_gd", "contrast")):
        return "t1ce"
    if "t2" in stem:
        return "t2"
    if "t1" in stem:
        return "t1"
    return None


def scan_name(filename: str) -> str:
    """The part of a file name that names the scan: BraTS2021_00110_t1ce.nii.gz -> 'brats2021_00110'."""
    words = [w for w in re.split(r"[_\-. ]+", filename.lower()) if w]
    return "_".join(w for w in words if w not in MODALITY_WORDS)


def different_scans_warning(choice: dict) -> str | None:
    """A warning when the file names suggest the four files come from different scans or people.

    The service can only refuse files whose image sizes differ; prepared scans from different people often have the
    same size, so the file names are the only clue. Files named only after the image type (t1.nii.gz) are skipped.
    """
    names = {choice[m].name: scan_name(choice[m].name) for m in MODALITIES if choice.get(m) is not None}
    if len({n for n in names.values() if n}) <= 1:
        return None
    listed = ", ".join(f"`{name}`" for name in names)
    return ("These files seem to come from different scans or different people: " + listed + ". "
            "All four files must be from the same scan, otherwise the results are wrong. "
            "Please check them before you continue.")


def choose_scan_files(key: str) -> dict:
    """One drop area for the four files of a scan; each file is matched to its MRI type, and the user can fix it."""
    uploaded = st.file_uploader(
        "Drop the four MRI files of this scan here (.nii or .nii.gz)",
        type=["nii", "gz"],
        accept_multiple_files=True,
        key=f"{key}_files",
        help="Your scan has four kinds of images (T1, T1ce, T2 and FLAIR). Select all four files at once.",
    )
    uploaded = [f for f in (uploaded or []) if f.name.lower().endswith((".nii", ".nii.gz"))]
    if not uploaded:
        return {}

    names = [f.name for f in uploaded]
    by_name = {f.name: f for f in uploaded}
    guesses = {m: next((n for n in names if guess_modality(n) == m), None) for m in MODALITIES}
    st.caption("Check that each file was matched to the right kind of image:")
    choice = {}
    columns = st.columns(len(MODALITIES))
    for column, modality in zip(columns, MODALITIES):
        short, meaning = MODALITY_INFO[modality]
        options = ["— not uploaded —"] + names
        default = options.index(guesses[modality]) if guesses[modality] else 0
        picked = column.selectbox(short, options, index=default, key=f"{key}_{modality}", help=meaning.capitalize())
        choice[modality] = by_name.get(picked)
    warning = different_scans_warning(choice)
    if warning:
        st.warning(warning)
    return choice


def check_scan_files(choice: dict, label: str = "") -> str | None:
    """A plain-language problem with the chosen files, or None when all four are there and different."""
    where = f" for {label}" if label else ""
    missing = [MODALITY_INFO[m][0] for m in MODALITIES if choice.get(m) is None]
    if missing:
        return f"Missing{where}: " + ", ".join(missing) + ". Please add the missing files."
    picked = [choice[m].name for m in MODALITIES]
    if len(set(picked)) != len(picked):
        return f"The same file was chosen twice{where}. Each of the four kinds of image needs its own file."
    return None


def files_payload(choice: dict, field=None) -> list:
    """Files for the API, in the order t1, t1ce, t2, flair."""
    return [
        (field or m, (choice[m].name, choice[m].getvalue(), "application/octet-stream"))
        for m in MODALITIES
    ]


def region_volume(volumes: dict, region: str) -> float | None:
    value = volumes.get(region, {})
    return value.get("volume_cm3") if isinstance(value, dict) else None


def describe_change(change: dict) -> str:
    """'grew by 1.2 cm³ (+15%)' / 'shrank by …' / 'stayed about the same' in plain words."""
    delta = change.get("change_cm3") or 0.0
    pct = change.get("change_percent")
    if abs(delta) < SAME_CM3:
        return "stayed about the same"
    word = "grew" if delta > 0 else "shrank"
    text = f"{word} by {abs(delta):.1f} cm³"
    if pct is not None:
        text += f" ({pct:+.0f}%)"
    else:
        text += " (too small before to give a percentage)"
    return text


def legend_markdown() -> str:
    swatch = '<span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:{};' \
             'margin:0 6px -2px 14px"></span>{}'
    return "".join(swatch.format(color, text) for color, text in LABEL_LEGEND)


def scan_viewer(images: dict, masks: dict, titles: dict, key: str, modality_choices=MODALITIES) -> None:
    """Slice viewer for one or more scans side by side; images[scan][modality] and masks[scan] are arrays."""
    scan_ids = list(masks)
    available = [m for m in modality_choices if all(m in images[s] for s in scan_ids)]
    control1, control2 = st.columns([1, 2])
    with control1:
        modality = st.selectbox(
            "Kind of image",
            available,
            index=available.index("flair") if "flair" in available else 0,
            format_func=lambda m: f"{MODALITY_INFO[m][0]}: {MODALITY_INFO[m][1]}",
            key=f"{key}_modality",
        )
    depth = min(masks[s].shape[2] for s in scan_ids)
    with control2:
        slice_index = st.slider(
            "Move through the brain (bottom ← → top)",
            min_value=0, max_value=depth - 1, value=depth // 2, step=1, key=f"{key}_slice",
        )
    show_mask = st.checkbox("Show the colored tumor outline", value=True, key=f"{key}_show")
    with st.expander("More display options"):
        opacity = st.slider("How strong the colors are", 0.10, 0.80, 0.45, 0.05, key=f"{key}_opacity")

    columns = st.columns(len(scan_ids))
    for column, scan_id in zip(columns, scan_ids):
        with column:
            if len(scan_ids) > 1:
                st.markdown(f"**{titles[scan_id]}**")
            image, mask = images[scan_id][modality], masks[scan_id]
            if image.shape != mask.shape:
                st.error("This scan and its tumor outline have different sizes, so they cannot be shown together.")
                continue
            fig = render_slice(image, mask, slice_index, opacity=opacity, show_mask=show_mask)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
            plt.close(fig)
    if show_mask:
        st.markdown(legend_markdown(), unsafe_allow_html=True)


def post(url: str, **kwargs):
    """POST to the API; returns the JSON reply or shows a plain-language error and returns None."""
    try:
        response = requests.post(url, timeout=(30, 3600), **kwargs)
    except requests.Timeout:
        st.error("The analysis took too long and was stopped. Please try again later.")
        return None
    except requests.RequestException as exc:
        st.error("The analysis service could not be reached. Please check that it is running.")
        with st.expander("Technical details"):
            st.code(str(exc))
        return None
    if not response.ok:
        st.error("The scan could not be analyzed. Check that the four files belong to the same scan "
                 "and are MRI files in .nii or .nii.gz format.")
        with st.expander("Technical details"):
            st.code(f"HTTP {response.status_code}\n{response.text[:5000]}")
        return None
    return response.json()


# ---------------------------------------------------------------------------
# Sidebar: technical settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Technical settings")
    api_base_url = st.text_input(
        "Analysis service address",
        value=API_DEFAULT,
        help="The address of the FastAPI backend. Leave it as it is unless you were told otherwise.",
    ).strip().rstrip("/")
    if st.button("Check the connection", use_container_width=True):
        try:
            health = requests.get(f"{api_base_url}/health", timeout=15)
            health.raise_for_status()
            if health.json().get("status") == "ok":
                st.success("The analysis service is ready.")
            else:
                st.warning("The analysis service answered, but it is not ready.")
            with st.expander("Details"):
                st.json(health.json())
        except requests.RequestException as exc:
            st.error(f"Could not reach the analysis service: {exc}")


# ---------------------------------------------------------------------------
# Introduction
# ---------------------------------------------------------------------------
st.title("🧠 Brain MRI Tumor Measurement")
st.write(
    "This tool looks at a brain MRI scan, outlines a brain tumor (glioma) in 3D and measures its size. "
    "It can also compare scans of the same person taken at different times."
)
st.warning(DISCLAIMER)

with st.expander("What files do I need? What are T1, T1ce, T2 and FLAIR?"):
    st.write(
        "An MRI scan for a brain tumor usually includes four kinds of images of the same brain. "
        "Each one shows the tumor differently, and the tool needs all four:"
    )
    st.dataframe(pd.DataFrame(
        [{"Kind of image": MODALITY_INFO[m][0], "What it shows": MODALITY_INFO[m][1].capitalize()} for m in MODALITIES]
    ), hide_index=True, use_container_width=True)
    st.write(
        "The files are usually named after the kind of image (for example `..._flair.nii.gz`), so the tool "
        "recognizes them by name. If it gets one wrong, you can change it before starting. "
        "The scans must already be prepared like the BraTS dataset (skull removed, aligned, 1 mm)."
    )

with st.expander("What does the tool measure?"):
    st.dataframe(pd.DataFrame(
        [{"Measurement": REGION_INFO[r][0], "Meaning": REGION_INFO[r][1].capitalize()} for r in ("WT", "TC", "ET")]
    ), hide_index=True, use_container_width=True)
    st.write("Sizes are given in cm³. 1 cm³ is 1 milliliter, about a fifth of a teaspoon.")

mode = st.radio(
    "What would you like to do?",
    ["Measure one scan", "Compare scans taken at different times"],
    horizontal=True,
)


# ---------------------------------------------------------------------------
# One scan
# ---------------------------------------------------------------------------
if mode == "Measure one scan":
    st.subheader("1. Add the scan")
    choice = choose_scan_files("single")

    if st.button("Measure the tumor", type="primary", use_container_width=True):
        problem = check_scan_files(choice)
        if problem:
            st.error(problem)
        else:
            case_id = f"case_{uuid.uuid4().hex[:12]}"
            with st.spinner("Analyzing the scan. This can take a few minutes..."):
                started = time.time()
                data = post(f"{api_base_url}/predict", files=files_payload(choice),
                            data={"case_id": case_id, "run_radiomics": "false"})
                elapsed = time.time() - started
            if data is not None:
                try:
                    result = data["result"]
                    mask_path = Path(result["segmentation"]["mask_path"])
                    with st.spinner("Preparing the pictures..."):
                        images = {m: load_uploaded_nifti(choice[m]) for m in MODALITIES}
                        mask = load_volume(str(mask_path))
                    st.session_state["single_analysis"] = {
                        "case_id": case_id, "result": result, "images": images, "mask": mask,
                        "elapsed": elapsed, "mask_path": str(mask_path),
                    }
                except (KeyError, ValueError, OSError) as exc:
                    st.error("The result could not be displayed.")
                    with st.expander("Technical details"):
                        st.code(f"{type(exc).__name__}: {exc}")

    analysis = st.session_state.get("single_analysis")
    if analysis:
        result = analysis["result"]
        volumes = result.get("segmentation", {}).get("volumes", {})
        st.divider()
        st.subheader("2. Measurements")
        columns = st.columns(3)
        for column, region in zip(columns, ("WT", "TC", "ET")):
            value = region_volume(volumes, region)
            name, meaning = REGION_INFO[region]
            column.metric(name, "–" if value is None else f"{value:.1f} cm³", help=meaning.capitalize())
            column.caption(meaning.capitalize())
        if (region_volume(volumes, "WT") or 0) == 0:
            st.info("The model did not outline any tumor on this scan.")
        st.caption("1 cm³ = 1 milliliter. These sizes describe the computer's outline, which can contain mistakes.")

        st.subheader("3. Look through the scan")
        scan_viewer({"scan": analysis["images"]}, {"scan": analysis["mask"]}, {"scan": "Scan"}, key="single")

        with st.expander("Technical details"):
            st.write("Reference:", analysis["case_id"])
            st.write(f"Time taken: {analysis['elapsed']:.1f} s")
            st.write("Mask file:", analysis["mask_path"])
            if result.get("inference_timings"):
                st.json(result["inference_timings"])
            if result.get("input_validation"):
                st.json(result["input_validation"])
    else:
        st.info("Add the four files of a scan and press **Measure the tumor**. The results will appear here.")


# ---------------------------------------------------------------------------
# Several scans of the same person
# ---------------------------------------------------------------------------
else:
    st.subheader("1. Add the scans, oldest first")
    st.write("Add every scan of the same person in the order they were taken. Give each one a name or a date.")
    n_scans = int(st.number_input("How many scans?", min_value=2, max_value=MAX_VISITS, value=2, step=1))

    scans = []
    for i in range(n_scans):
        with st.container(border=True):
            name = st.text_input(
                f"Scan {i + 1}: name or date",
                value="",
                placeholder="for example: March 2025, or before surgery",
                key=f"visit_name_{i}",
            ).strip() or f"Scan {i + 1}"
            scans.append((name, choose_scan_files(f"visit_{i}")))

    if st.button("Compare the scans", type="primary", use_container_width=True):
        names = [name for name, _ in scans]
        problems = [check_scan_files(choice, name) for name, choice in scans]
        problems = [p for p in problems if p]
        if problems:
            for problem in problems:
                st.error(problem)
        elif len(set(names)) != len(names):
            st.error("Each scan needs a different name.")
        else:
            payload = [item for _, choice in scans for item in files_payload(choice, field="files")]
            visits_json = json.dumps([{"visit_id": f"scan_{i + 1}"} for i in range(len(scans))])
            with st.spinner("Analyzing every scan. This can take a few minutes per scan..."):
                data = post(f"{api_base_url}/predict/longitudinal", files=payload,
                            data={"patient_id": f"person_{uuid.uuid4().hex[:12]}", "visits_json": visits_json,
                                  "run_radiomics": "false"})
            if data is not None:
                try:
                    images, masks, titles = {}, {}, {}
                    with st.spinner("Preparing the pictures..."):
                        for (name, choice), visit in zip(scans, data["visits"]):
                            visit_id = visit["visit_id"]
                            titles[visit_id] = name
                            images[visit_id] = {"flair": load_uploaded_nifti(choice["flair"]),
                                                "t1ce": load_uploaded_nifti(choice["t1ce"])}
                            masks[visit_id] = load_volume(visit["segmentation"]["mask_path"])
                    st.session_state["compare_analysis"] = {"data": data, "images": images, "masks": masks,
                                                            "titles": titles}
                except (KeyError, ValueError, OSError) as exc:
                    st.error("The results could not be displayed.")
                    with st.expander("Technical details"):
                        st.code(f"{type(exc).__name__}: {exc}")

    analysis = st.session_state.get("compare_analysis")
    if analysis:
        data, titles = analysis["data"], analysis["titles"]
        st.divider()
        st.subheader("2. Size of each area in every scan")
        rows = []
        for visit in data.get("visits", []):
            volumes = visit.get("segmentation", {}).get("volumes", {})
            row = {"Scan": titles.get(visit["visit_id"], visit["visit_id"])}
            for region in ("WT", "TC", "ET"):
                value = region_volume(volumes, region)
                row[f"{REGION_INFO[region][0]} (cm³)"] = None if value is None else round(value, 1)
            rows.append(row)
        table = pd.DataFrame(rows)
        st.dataframe(table, hide_index=True, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        for column in table.columns[1:]:
            ax.plot(range(len(table)), table[column], marker="o", label=column.replace(" (cm³)", ""))
        ax.set_xticks(range(len(table)), table["Scan"])
        ax.set_ylabel("Size (cm³)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        plt.close(fig)

        st.subheader("3. What changed")
        for change in data.get("longitudinal_changes", []):
            if change.get("region") not in REGION_INFO:
                continue
            before = titles.get(change["from_visit"], change["from_visit"])
            after = titles.get(change["to_visit"], change["to_visit"])
            st.markdown(f"- **{REGION_INFO[change['region']][0]}**, from *{before}* to *{after}*: "
                        f"{describe_change(change)}")
        st.caption("Small changes can come from differences between scans or from the computer's outline, "
                   "not only from the tumor itself. Ask your doctor what a change means for you.")

        st.subheader("4. See the scans side by side")
        scan_viewer(analysis["images"], analysis["masks"], titles, key="compare", modality_choices=("flair", "t1ce"))

        with st.expander("Technical details"):
            st.json(data.get("longitudinal_changes", []))
    else:
        st.info("Add every scan and press **Compare the scans**. The results will appear here.")
