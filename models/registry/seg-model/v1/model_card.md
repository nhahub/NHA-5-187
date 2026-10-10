# Model card: seg-model v1 (Selected Model, end of Milestone 2)

| | |
| --- | --- |
| Model | nnU-Net v2, `3d_fullres`, fold 0, trainer `nnUNetTrainer_50epochs` (50 epochs x 250 steps) |
| Task | Brain tumor segmentation from MRI: three nested regions TC (tumor core), WT (whole tumor), ET (enhancing tumor) |
| Input | 4 MRI types per patient, in this channel order: T1, T1ce, T2, FLAIR (cleaned scans from M1-T04) |
| Output | Label map: 0 background, 1 NCR/NET, 2 edema, 3 enhancing tumor |
| Source task | M2-T04 (nnU-Net vs 3D U-Net comparison), notebook `notebooks/M2_T04_nnUNet_vs_3D_UNet2.ipynb` |
| Weights | `checkpoint_best.pth` (about 250 MB, 249,741,887 bytes), kept on Drive, not in git. SHA-256 in `manifest.json` |
| Intended use | Research prototype for this project. **Not a medical device and not validated for clinical use.** |

## Data used

- **BraTS 2021**, cleaned in M1-T04 (adult glioma, pre-treatment scans, expert segmentations).
- Split `split_v1` (seed 42): **875 training** patients, **188 validation** patients.
- The **188 locked test patients were never read** while training or choosing the model.
- Other datasets in the project (RHUH-GBM, MU-Glioma-Post, BraTS-Africa, UTSW-Glioma) were **not** used for this model.

## Scores (188 validation patients)

| Region | Dice | IoU | Sensitivity | Precision | HD95 (mm) |
| --- | --- | --- | --- | --- | --- |
| TC | 0.917 | 0.867 | 0.905 | 0.946 | 4.60 |
| WT | 0.935 | 0.882 | 0.917 | 0.960 | 5.62 |
| ET | 0.872 | 0.802 | 0.877 | 0.896 | 4.29 |
| **Mean Dice** | **0.908** | | | | |

Baseline for comparison, 3D U-Net (EXP-0001): mean Dice 0.765. nnU-Net scored higher on 97% of validation patients (paired Wilcoxon p = 3.2e-31).

## Known limits

- **Validation scores, not test scores.** The checkpoint was picked by nnU-Net's own pseudo Dice on the validation fold, so the validation numbers are somewhat optimistic. The locked test set has not been used yet.
- **One split, one run.** No repeated runs or cross-validation, so there is no confidence interval.
- **Short training.** 50 epochs instead of nnU-Net's default 1000; the model was still improving, so this is probably not its ceiling.
- **Narrow data.** Trained only on BraTS 2021 (pre-treatment adult glioma). Post-treatment scans, other tumor types, children, other scanners or protocols have not been tested. Performance there is unknown and likely lower.
- **Needs all 4 MRI types** in the exact order above, already cleaned the same way as in M1-T04.
- **False enhancing tumor in 3 patients.** 3 validation patients (BraTS2021_01483, 01509, 01535) have no enhancing tumor in the expert mask, but the model predicted some. Their ET Dice is 0 and is included in the ET average (0.872; 0.886 without them). Their ET HD95 and sensitivity are undefined, so those averages leave them out. The model can mark enhancing tumor where there is none.
- **Inference uses mirroring test-time augmentation**, which makes it slower than a single pass.
- **Not tracked in MLflow.** The nnU-Net run has no MLflow run id; its scores are in `results/M2-T04/`.

## Where things are

- Scores and comparison: `docs/TRAINING_NNUNET.md`, `results/M2-T04/`
- Weights and predictions (Drive): `BrainMRI_Data/runs/M2-T04_nnunet/`
- How to run it: `inference_config.yaml`; checksum and source: `manifest.json`
  
## Approval (M2-T06)

| Name | Role | OK (date) |
| --- | --- | --- |
| | | |
