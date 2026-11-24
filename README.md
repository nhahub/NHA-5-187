# AI-Powered Brain MRI Tumor Analysis, Segmentation, and Quantitative Assessment System

DEPI (Digital Egypt Pioneers Initiative) — AI & Data Science track, Machine Learning graduation project, 2026.
Instructor: Eng. Aya Abdallah. Team: Mohamed Tagy, Alaa Emad, Hagar Aiman, Yasmin Mahmoud, Maryam Ahmed, Essam Mohamed.

The system takes the four MRI types of a brain scan (T1, T1ce, T2, FLAIR), finds the tumor in 3D, and reports its three
regions: **tumor core (TC)**, **whole tumor (WT)** and **enhancing tumor (ET)**. Later milestones add tumor volumes,
change over time, radiomics, a confidence map and a web app.

> **Decision-support research prototype only.** It is not a medical device, it does not diagnose anyone, and it has
> not been validated for clinical use.

## Status (23 Nov 2026)

| Milestone | State |
| --- | --- |
| M1 – Data foundation | Done: datasets downloaded, every file checksummed, scans cleaned, patients split (locked test set), [EDA report](docs/EDA_REPORT.md) |
| M2 – Segmentation model | Done: nnU-Net selected and registered with an approved [model card](models/registry/seg-model/v1/model_card.md) |
| M3 – Analysis (volumes, change over time, radiomics, confidence map) | Done: volumes ([`results/M3-T01`](results/M3-T01/README.md)), change over time ([`results/M3-T02`](results/M3-T02/README.md)), radiomics ([`results/M3-T03`](results/M3-T03/README.md)), [confidence maps](docs/UNCERTAINTY.md), all in one [analysis pipeline](docs/PIPELINE.md) |
| M4 – App and deployment | In progress: seg-model v1 [packaged for fast prediction](docs/DEPLOY_MODEL.md) (2.1 s per scan on a T4; [`results/M4-T01`](results/M4-T01/README.md)), FastAPI backend (`api/`), web interface for one scan or several visits (`apps/web_interface_m4_t03.py`) and Docker (`docker-compose.yml`) done; Azure deployment ([`deploy/azure.md`](deploy/azure.md)) and the app test in progress |
| M5 – Final tests and delivery | Not started |

Scores on the 188 BraTS 2021 validation patients (the locked test patients have not been used yet):

| Model | Dice TC | Dice WT | Dice ET | Mean Dice | Details |
| --- | --- | --- | --- | --- | --- |
| 3D U-Net baseline (EXP-0001) | 0.717 | 0.848 | 0.728 | 0.765 | [`docs/TRAINING_BASELINE.md`](docs/TRAINING_BASELINE.md) |
| **nnU-Net 3d_fullres (selected, seg-model v1)** | **0.917** | **0.935** | **0.872** | **0.908** | [`docs/TRAINING_NNUNET.md`](docs/TRAINING_NNUNET.md), [model card](models/registry/seg-model/v1/model_card.md) |

These are validation scores of a research prototype on a public dataset, not evidence of clinical performance.

**Project documentation** (planning, stakeholder analysis, data design, UI/UX design): [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md).

## Data

Training uses **BraTS 2021** only; RHUH-GBM, MU-Glioma-Post, BraTS-Africa and UTSW-Glioma are kept for external tests
and the change-over-time module. Reasons, licenses and download steps: [`docs/dataset_research.md`](docs/dataset_research.md),
[`docs/DATA_DOWNLOAD.md`](docs/DATA_DOWNLOAD.md), [`docs/DATASETS_M1_T01.md`](docs/DATASETS_M1_T01.md).
What the data looks like (cohort sizes, tumor sizes and composition, class imbalance, tumor location, intensities,
difficult cases, with charts): [`docs/EDA_REPORT.md`](docs/EDA_REPORT.md), made by `notebooks/06_eda.ipynb`.

**No MRI data, model weights or keys are stored in this repository** (it is public). The data and checkpoints live on
the team's Google Drive; the repository keeps only code, configs, small result files and the patient lists with checksums
(`data/manifests/`).

## Repository layout

| Folder | Content |
| --- | --- |
| `src/bmts/` | Python package: scan cleaning (`data/cleaning`), data loader (`data/datasets`, `data/augmentation`), metrics (`evaluation`), 3D U-Net model and training loop (`segmentation`), split rules (`data/splits.py`), volumes and change over time (`quantitative`), radiomics (`radiomics`), confidence maps (`uncertainty`), the analysis pipeline (`pipeline`), the exported model and fast segmenter (`deploy`) |
| `configs/` | YAML settings for data, models and experiments (`training/EXP-0001.yaml`) |
| `scripts/` | Command-line tools: `download_data.py`, `smoke_train.py`, `train.py`, `eda.py`, `uncertainty.py`, `analyze.py` (the whole analysis for one patient), `export_model.py` and `segment.py` (fast prediction) |
| `notebooks/` | Colab notebooks, numbered in the order they are used (download → patient list → loader test → training → EDA → M3 tasks → `08_m3_t05_analysis_pipeline` → `09_m4_t01_export_model` → `10_m4_t06_demo_cases`, which puts ready-to-upload test scans in `BrainMRI_Data/demo_cases/`), plus the M2 comparison and viewer notebooks |
| `apps/` | MRI slice viewer with the tumor mask on top (Streamlit) |
| `models/registry/` | Registered model versions: model card, manifest, inference settings, nnU-Net plans (weights on Drive) |
| `results/` | Small result files: EDA tables and charts (`eda/`), the M2-T04 model comparison, one folder per M3 task, the M4-T01 speed check |
| `data/manifests/` | Patient list and per-file SHA-256 checksums of every dataset |
| `docs/` | Project structure, dataset research, how-to guides and results |
| `tests/` | Automated tests (`pytest tests`) |
| `Reports/` | Milestone reports |

## Running it

Most work runs in **Google Colab** (free T4 GPU) against the shared Drive folder `BrainMRI_Data`: open a notebook from
`notebooks/` with **File → Open notebook → GitHub**, choose a GPU runtime, and **Run all**. Locally:

```bash
pip install -r environment/requirements-dev.txt -r environment/requirements-viewer.txt
pytest tests                                        # loader, training and viewer tests
python scripts/train.py --exp configs/training/EXP-0001.yaml --help
python scripts/analyze.py --help                    # mask, volumes, change, radiomics, confidence map for one patient
python scripts/segment.py --help                    # fast tumor mask with the exported model, see docs/DEPLOY_MODEL.md
streamlit run apps/m2_t05_viewer.py                 # needs cleaned scans, see docs/M2_T05_VIEWER.md
```

## Working on the code

The initiative does not allow adding collaborators, so contributions come as pull requests from forks:

1. **Fork** this repository, clone your fork and add the main one: `git remote add upstream https://github.com/nhahub/NHA-5-187.git`.
2. For each task: `git checkout main`, `git pull upstream main`, then a branch named after the task ID, e.g. `M3-T04-confidence-map`.
3. Commit messages start with the task ID; push to your fork and open a pull request into `main`. Mohamed reviews and merges.
4. Never commit data, model checkpoints, `.env` files or credentials. Keep tokens in Colab Secrets.

Task board (tasks, owners, deadlines): the team's Notion board. Repository conventions: [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md).
