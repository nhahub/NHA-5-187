Final test plan (M5-T01)

Status: DRAFT until the lock section at the bottom is filled in. After the lock, nobody changes the rules below. Any later change is written in the "Amendments" table, with the date and the reason, and the original text stays as it is.


This plan is written before any result on the locked test data has been seen.

1. What is tested
Item	Value
Model	seg-model v1 (nnU-Net 3d_fullres, trainer nnUNetTrainer_50epochs, fold 0)
Checkpoint	checkpoint_best.pth, SHA-256 7dcb450bf5455a0c62c9e1b368ee6341c687f555528d5ed3cd3399d88ea0f5e0 (the value in models/registry/seg-model/v1/manifest.json)
Code version	commit hash: [FILL IN at lock time]

The model is not retrained, tuned or changed after this plan is locked.

2. Which data
Group	Data	Size	Used for
A. Locked test	BraTS 2021 patients with split = test in BrainMRI_Data/qc/split_v1.csv	188 patients (split 875 / 188 / 188, fixed seed)	Main score
B. External A	UTSW-Glioma (split = external)	625 patients	Does the model work in another hospital?
C. External B	BraTS-Africa (split = external)	146 patients	Does the model work on another population?
D1. Post-treatment	MU-Glioma-Post (split = test)	30 patients, all their scans	Segmentation score on post-treatment scans
D2. Longitudinal	RHUH-GBM (split = test)	6 patients, all their scans (very small, read with care)	Segmentation score on pre- and post-operative scans

The patient numbers come from BrainMRI_Data/qc/split_v1.csv. The manifests in the repository list one more patient for UTSW (626) and for Africa (147); the split file is the one that counts. The split file has 1,087 locked rows. That is 188 + 146 + 625 = 959 patients plus about 128 scans of the 36 MU and RHUH test patients. The exact number of scans is printed when the test is run.

Rules:

Only patients marked locked in split_v1.csv (test, external or lockbox = True) are used here. They were never used for training, validation or tuning.
Patients who are in BraTS 2021 are not allowed in the external groups (data/manifests/unified_subject_index.csv is used to check this).
Each group is scored on its own. The groups are never merged into one number. For D1 and D2 every scan that has an expert mask is scored, and we report both the number of patients and the number of scans.
The MU-Glioma-Post and RHUH-GBM patients with split = train or val were used for development, so they are not part of this test. All scans of one patient are always in the same group.
The validation patients (188) and the training patients are not part of this test.
3. Which scores

For each patient and for each of the three regions:

Region	Labels
WT (whole tumor)	1, 2, 3
TC (tumor core)	1, 3
ET (enhancing tumor)	3
Dice (higher is better, 1 = perfect).
HD95 (95th percentile Hausdorff distance, in mm, lower is better).
Both are computed with the project functions in src/bmts/evaluation (already checked against MONAI in M2-T02).

Empty regions (decided now, before the test):

Both masks empty: Dice = 1 and HD95 = 0.
Only one mask empty: Dice = 0 and HD95 is nan.
The average of HD95 ignores nan, so we also report how many patients had nan, per region and per group. These are cases where the model missed a region or invented one, and they must not stay hidden.

What we report for every group: mean and standard deviation of Dice and HD95 for WT, TC and ET, the number of patients, the number of nan cases, and the 5 worst patients per region (patient ID and score).

4. How many times
The final test is run once, by M5-T02, with allow_lockbox=True.
If it crashes or the numbers are clearly wrong because of a bug (not because the scores are low), it may be re-run after the bug is fixed. This is written in the Amendments table with the reason.
Low scores are not a reason to change the model or the plan and run again.
5. What counts as a good result

Fixed now, before any result is seen. The margins below are our own choice, not a published standard.

Group	Pass rule
A (locked BraTS test)	Mean Dice of each region (WT, TC, ET) is at most 0.03 below its validation value (WT 0.935, TC 0.917, ET 0.872), and the mean of the three is above the 3D U-Net baseline (0.765)
B, C, D1, D2 (external, post-treatment, longitudinal)	No pass or fail. We report the scores and the gap to group A. Group D2 has only 6 patients, so it is shown as a table of single results. We expect them to be lower, and a lower score is a finding, not a failure.

For reference: validation scores come from the 188 validation patients. The checkpoint was selected using the validation fold, so the validation numbers are somewhat optimistic.

6. Not part of this plan

Volumes, radiomics, confidence maps and the web app are tested in their own tasks and are not pass or fail here.

7. Lock
Item	Value
Date locked	[FILL IN]
Plan commit hash on main	[FILL IN after merge]
SHA-256 of split_v1.csv	a11d0ab1b19ebfa9845970b2532f50a184da43ec91fbda2c4a640c1cee5acedc
Approved by	Mohamed Tagy [date]
8. Amendments (after the lock)
Date	What changed	Why	Approved by
			
