"""Web interface tests (M4-T03): the page starts without errors, files are matched to their MRI type by name,
and changes are described in plain words."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "apps" / "web_interface_m4_t03.py"
sys.path.insert(0, str(REPO))          # the page imports apps.m2_t05_viewer


@pytest.fixture(scope="module")
def page():
    """The script run once by Streamlit's test runner; its functions are then read from the script's namespace."""
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    return app


def helpers():
    namespace = {"__name__": "web_interface_helpers"}
    source = APP.read_text(encoding="utf-8")
    exec(compile(source.split("# Sidebar: technical settings")[0], str(APP), "exec"), namespace)
    return namespace


def test_page_starts_with_the_disclaimer_and_both_modes(page):
    assert not page.exception
    assert any("Always discuss the results with your doctor" in w.value for w in page.warning)
    assert page.radio[0].options == ["Measure one scan", "Compare scans taken at different times"]


def test_compare_mode_starts_without_errors(page):
    page.radio[0].set_value("Compare scans taken at different times").run()
    assert not page.exception
    assert any(b.label == "Compare the scans" for b in page.button)


@pytest.mark.parametrize("name, expected", [
    ("BraTS2021_00110_t1.nii.gz", "t1"),
    ("BraTS2021_00110_t1ce.nii.gz", "t1ce"),
    ("BraTS2021_00110_t2.nii.gz", "t2"),
    ("BraTS2021_00110_flair.nii.gz", "flair"),
    ("patient_T1Gd.nii", "t1ce"),
    ("t2_flair.nii.gz", "flair"),
    ("scan.nii.gz", None),
])
def test_files_are_matched_by_name(name, expected):
    assert helpers()["guess_modality"](name) == expected


def test_changes_are_described_in_plain_words():
    describe = helpers()["describe_change"]
    assert describe({"change_cm3": 1.234, "change_percent": 15.2}) == "grew by 1.2 cm³ (+15%)"
    assert describe({"change_cm3": -2.0, "change_percent": None}).startswith("shrank by 2.0 cm³ (too small before")
    assert describe({"change_cm3": 0.04, "change_percent": 1.0}) == "stayed about the same"


def test_missing_or_repeated_files_get_a_clear_message():
    h = helpers()
    check = h["check_scan_files"]

    class File:
        def __init__(self, name):
            self.name = name

    files = {m: File(f"{m}.nii.gz") for m in h["MODALITIES"]}
    assert check(files) is None
    assert "Missing" in check({**files, "t2": None}) and "T2" in check({**files, "t2": None})
    assert "twice" in check({**files, "t2": files["t1"]})


@pytest.mark.parametrize("name, expected", [
    ("BraTS2021_00110_t1ce.nii.gz", "brats2021_00110"),
    ("RHUH-0002_0_flair.nii.gz", "rhuh_0002_0"),
    ("patient_T1Gd.nii", "patient"),
    ("t2.nii.gz", ""),
])
def test_scan_name_drops_the_image_type(name, expected):
    assert helpers()["scan_name"](name) == expected


def test_files_from_different_scans_get_a_warning():
    h = helpers()
    warn = h["different_scans_warning"]

    class File:
        def __init__(self, name):
            self.name = name

    same = {m: File(f"BraTS2021_00110_{m}.nii.gz") for m in h["MODALITIES"]}
    assert warn(same) is None
    mixed = {**same, "flair": File("BraTS2021_00111_flair.nii.gz")}
    assert "different scans" in warn(mixed) and "BraTS2021_00111_flair.nii.gz" in warn(mixed)
    plain = {m: File(f"{m}.nii.gz") for m in h["MODALITIES"]}
    assert warn(plain) is None                                        # no scan name in the file names: no clue
