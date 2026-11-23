"""M2-T05: simple Brain MRI slice viewer with tumor-mask overlay.

Run locally:
    streamlit run apps/m2_t05_viewer.py

The viewer uses the cleaned M1-T04 layout:
    <processed_root>/<dataset>/<subject>/<timepoint>/{t1,t1ce,t2,flair,seg}.nii.gz

This task intentionally visualizes the existing segmentation mask (seg.nii.gz).
It does NOT run the trained U-Net; M2-T05 depends on the cleaned scans from M1-T04,
not on M2-T03 model inference.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap, to_rgba

MODALITIES = ("t1", "t1ce", "t2", "flair")
SEG = "seg"
REQUIRED_FILES = (*MODALITIES, SEG)
LABEL_NAMES = {
    0: "Background",
    1: "NCR/NET",
    2: "Edema",
    3: "Enhancing tumor",
    4: "Resection cavity",
}
# Same colors as the EDA and confidence-map figures (docs/PROJECT_DOCUMENTATION.md, section 4.2)
LABEL_COLORS = {
    1: "#eb6834",  # orange: necrotic core
    2: "#1baf7a",  # green: edema
    3: "#e87ba4",  # pink: enhancing tumor
    4: "#eda100",  # amber: resection cavity
}


@dataclass(frozen=True)
class Case:
    dataset: str
    subject: str
    timepoint: str
    directory: Path

    @property
    def id(self) -> str:
        return f"{self.dataset}/{self.subject}/{self.timepoint}"


def _nifti_path(case: Case, name: str) -> Path:
    return case.directory / f"{name}.nii.gz"


def discover_cases(processed_root: str | Path) -> list[Case]:
    """Find complete cleaned cases, even when the user points at a parent folder.

    The canonical layout is:
        root/dataset/subject/timepoint/{t1,t1ce,t2,flair,seg}.nii.gz

    We recursively inspect ``root`` so this also works when the actual processed
    folder is nested under a project/Drive folder. Raw BraTS folders are ignored
    because they use subject-specific filenames rather than the cleaned names.
    """
    root = Path(processed_root).expanduser()
    if not root.is_dir():
        return []

    cases: list[Case] = []
    seen: set[Path] = set()
    for tp_dir in sorted(root.rglob("seg.nii.gz"), key=lambda p: str(p).lower()):
        tp_dir = tp_dir.parent
        case_files = {name: tp_dir / f"{name}.nii.gz" for name in REQUIRED_FILES}
        if not all(path.is_file() for path in case_files.values()):
            continue
        if tp_dir in seen or len(tp_dir.parents) < 3:
            continue

        # By convention the three parents are subject, dataset, and root-like
        # container. We use the two immediate folder names for the case ID.
        subject_dir = tp_dir.parent
        dataset_dir = subject_dir.parent
        case = Case(dataset_dir.name, subject_dir.name, tp_dir.name, tp_dir)
        cases.append(case)
        seen.add(tp_dir)

    return cases


@st.cache_data(show_spinner=False)
def load_volume(path_str: str) -> np.ndarray:
    """Load a NIfTI file as a canonical RAS array."""
    img = nib.as_closest_canonical(nib.load(path_str))
    data = np.asanyarray(img.dataobj)
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D NIfTI volume: {path_str}; got shape {data.shape}")
    return np.asarray(data)


def validate_case(case: Case) -> tuple[int, int, int]:
    """Ensure all modalities and the mask share the same voxel grid shape."""
    shapes = {name: tuple(load_volume(str(_nifti_path(case, name))).shape) for name in REQUIRED_FILES}
    unique = set(shapes.values())
    if len(unique) != 1:
        raise ValueError(f"Mismatched volume shapes in {case.id}: {shapes}")
    return next(iter(unique))


def normalize_for_display(volume: np.ndarray) -> np.ndarray:
    """Robustly map an MRI volume to [0, 1] while ignoring zero background."""
    values = volume[np.isfinite(volume)]
    nonzero = values[np.abs(values) > 1e-8]
    sample = nonzero if nonzero.size >= 16 else values
    if sample.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(sample, (1, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(sample)), float(np.max(sample))
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0, 1).astype(np.float32)


def render_slice(
    image: np.ndarray,
    mask: np.ndarray,
    slice_index: int,
    opacity: float = 0.45,
    show_mask: bool = True,
) -> plt.Figure:
    """Render one axial slice with a discrete tumor-mask overlay."""
    if image.shape != mask.shape:
        raise ValueError(f"MRI/mask shapes differ: {image.shape} vs {mask.shape}")
    if not 0 <= slice_index < image.shape[2]:
        raise IndexError(f"slice_index {slice_index} out of range [0, {image.shape[2] - 1}]")

    base = np.rot90(normalize_for_display(image[:, :, slice_index]))
    mask_slice = np.rot90(np.rint(mask[:, :, slice_index]).astype(np.int16))

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.imshow(base, cmap="gray", interpolation="nearest")
    if show_mask:
        # Discrete colors: 0 transparent, 1..4 tumor labels.
        colors = [(0, 0, 0, 0)] + [to_rgba(LABEL_COLORS[label], opacity) for label in (1, 2, 3, 4)]
        cmap = ListedColormap(colors)
        ax.imshow(mask_slice, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    return fig


def _legend_text(mask: np.ndarray) -> str:
    present = sorted(int(x) for x in np.unique(mask) if int(x) in LABEL_NAMES and int(x) != 0)
    if not present:
        return "No tumor-mask labels are present in this slice."
    return "  |  ".join(f"{LABEL_NAMES[x]} ({LABEL_COLORS[x]})" for x in present)


def _default_root() -> str:
    """Pick a useful default across Colab, local project, and Windows setups."""
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        os.environ.get("BMTS_PROCESSED_ROOT", ""),
        "/content/processed",
        str(project_root / "data" / "processed"),
        str(Path.cwd() / "data" / "processed"),
        str(Path.cwd()),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir() and discover_cases(candidate):
            return candidate
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return candidate
    return str(project_root / "data" / "processed")


def main():
    """The Streamlit page. Runs only when launched with `streamlit run apps/m2_t05_viewer.py`,
    so the helper functions above can be imported (e.g. by tests) without starting the page."""
    st.set_page_config(page_title="Brain MRI Viewer — M2-T05", layout="wide")
    st.title("Brain MRI Viewer — M2-T05")
    st.caption("MRI slice viewer with the existing tumor segmentation mask overlaid. Decision-support/research prototype only.")

    with st.sidebar:
        st.header("Viewer settings")
        root = st.text_input("Processed data root", value=_default_root(), help="Folder containing dataset/subject/timepoint folders.")
        opacity = st.slider("Mask opacity", min_value=0.10, max_value=0.80, value=0.45, step=0.05)
        show_mask = st.checkbox("Show tumor mask", value=True)

    cases = discover_cases(root)
    if not cases:
        st.error(
            "No complete cleaned cases were found. Expected files: "
            "<root>/<dataset>/<subject>/<timepoint>/{t1,t1ce,t2,flair,seg}.nii.gz"
        )
        st.info(
            "This viewer does not create the cleaned data. M2-T05 depends on M1-T04. "
            "Point 'Processed data root' to the folder that contains the cleaned NIfTI files. "
            "The viewer can search nested folders automatically."
        )
        if Path(root).is_dir():
            segs = list(Path(root).rglob("seg.nii.gz"))
            if segs:
                st.warning(
                    f"Found {len(segs)} seg.nii.gz file(s), but none had all five required files "
                    "(t1, t1ce, t2, flair, seg) in the same folder."
                )
                st.code("\n".join(str(p.parent) for p in segs[:10]))
            else:
                st.warning("No seg.nii.gz file was found under this folder. You likely need the M1-T04 processed data.")
        else:
            st.warning(f"The selected root does not exist: {root}")
        st.stop()

    labels = [case.id for case in cases]
    selected_id = st.selectbox("Patient / timepoint", labels)
    case = cases[labels.index(selected_id)]

    try:
        shape = validate_case(case)
    except Exception as exc:  # noqa: BLE001 - surface a useful UI error
        st.error(f"Could not load {case.id}: {exc}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        modality = st.selectbox("MRI modality", list(MODALITIES), index=3)
    with col2:
        slice_index = st.slider("Axial slice", 0, shape[2] - 1, shape[2] // 2, 1)
    with col3:
        st.metric("Volume", f"{shape[0]} × {shape[1]} × {shape[2]}")

    image = load_volume(str(_nifti_path(case, modality)))
    mask = load_volume(str(_nifti_path(case, SEG)))
    fig = render_slice(image, mask, slice_index, opacity=opacity, show_mask=show_mask)
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)

    st.caption(_legend_text(mask[:, :, slice_index]))

    with st.expander("Case files"):
        for name in REQUIRED_FILES:
            st.code(str(_nifti_path(case, name)))


if __name__ == "__main__":
    main()
