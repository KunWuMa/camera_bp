# Data preparation and signal-processing protocol

## Acquisition-level inputs

The private pipeline pairs a facial video, optional fingertip PPG, acquisition
metadata, and acquisition-level SBP/DBP labels. Video and PPG were recorded
synchronously. The camera protocol used a Logitech C930 at 30 frames/s and
1920 x 1080 pixels. Contact PPG was used only for teacher/ablation experiments;
camera-model inference uses rPPG alone.

See `data/private/README.md` for access restrictions, local layout, and label
schema. See `data/public/rppg_bp_ukl/README.md` for the public release.

## Stage A: manifest and folds

`src/build_manifest.py`:

1. discovers private acquisitions and matches labels to modalities;
2. creates secret-salted 16-character SHA-256 subject and session pseudonyms;
3. selects a canonical video when multiple compression variants exist;
4. creates deterministic group-preserving folds; and
5. verifies that no subject occurs in both train and test for the stored strict
   assignment.

Outputs:

- `artifacts/manifests/session_manifest.csv`
- `artifacts/manifests/manifest_summary.csv`

These outputs are sensitive derived data and must not be committed.

## Stage B: facial RGB extraction

`src/extract_signals.py` downsamples frames by two for face detection. An OpenCV
Haar frontal-face detector uses scale factor 1.12, five neighbors, and minimum
80 x 80 pixels; detection is refreshed every 60 frames and the last box is reused
between detections.

Four fractional face regions are extracted:

- forehead: x 22--78%, y 10--30%;
- left cheek: x 10--42%, y 48--75%;
- right cheek: x 58--90%, y 48--75%;
- extended facial skin: x 10--90%, y 8--82%.

A YCrCb skin mask retains Y > 25, 125 < Cr < 180, and 70 < Cb < 140. If fewer
than 50 pixels or 5% of the crop survive, all crop pixels are used. For each RGB
channel, values below the 2nd and above the 98th spatial percentile are removed
before averaging.

Output files are per-acquisition compressed NumPy archives under
`artifacts/signals/<cohort>/`. They must never be uploaded.

## Stage C: rPPG construction and candidate selection

`src/prepare_dataset.py` resamples RGB traces to 30 Hz and creates POS and CHROM
candidates for every ROI. POS uses 1.6-second windows with overlap-add. Each of
the eight ROI/projection candidates is linearly detrended and passed through a
third-order 0.7--4.0 Hz Butterworth band-pass filter using zero-phase SOS
filtering.

Welch power spectral density is used for label-free candidate selection. The
largest peak in 0.7--4.0 Hz is located; signal power is integrated within
+/-0.1 Hz of the peak and, when in band, its second harmonic. Noise is the
remaining in-band power. The maximum-SNR candidate is selected without BP,
contact-PPG, or labeled-HR information.

## Stage D: windows and features

Private signals are divided into 10-second windows at 5-second stride. Every
window is independently z-scored. The generated `windows.h5` contains:

- `face`, `finger`: waveform windows;
- `labels`: SBP, DBP, and HR (HR may be missing and its loss is masked);
- `quality`: face SNR, fingertip SNR, and face-detection rate;
- `demographics`: age, encoded sex, height, and weight;
- `subject_id`, `session_id`, `source`, and `fold`.

The pipeline also writes `session_features.csv` with time-domain, spectral, peak,
and inter-beat descriptors used by classical models and the identity audit.

## Validation

`src/validate_artifacts.py` checks missing signals, finite labels/features, unique
acquisitions, and fold-level subject overlap. Stop the pipeline if any leakage or
data-integrity check fails. Do not repair invalid rows after viewing test results;
corrections require a new governed data snapshot and a complete rerun.
