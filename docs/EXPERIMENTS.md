# Experiment specification

## Primary question

How much does evaluation grouping change apparent camera-rPPG blood-pressure
performance, and does the effect persist across model families?

The primary endpoint is subject-macro MAE for SBP and DBP. Acquisition/window
pooled MAE is secondary.

## Evaluation protocols

- **Random-window**: windows are randomly assigned to folds. Adjacent windows and
  subject identities may overlap across train and test.
- **Random-session**: complete private acquisitions remain intact, but identities
  may overlap. This protocol applies only to the two private cohorts.
- **Subject-disjoint**: all acquisitions/windows belonging to a test subject are
  excluded from training and validation.

Private protocols use five outer folds. The stored laboratory subject fold is
leave-one-subject-out for the baseline processing, while the multi-protocol audit
maps evaluation to five comparable folds. The public cohort uses four folds.
Validation subjects are selected only from the outer-training population.

## Confirmatory neural architectures

- **Face-ANN**: three-layer temporal convolutional encoder plus bidirectional GRU.
- **Face-SNN**: the same convolutional stem followed by a recurrent LIF layer,
  eight simulation steps, beta 0.9, trainable threshold, and surrogate gradient.
- **ResNet-1D**: compact residual one-dimensional convolutional regressor.
- **TCN**: causal temporal blocks with dilation 1, 2, 4, and 8.
- **Transformer**: convolutional tokens, learned classification token, sinusoidal
  positions, two encoder layers, four attention heads.

Private models have three outputs: SBP, DBP, and auxiliary HR. When HR is missing,
only the HR loss element is masked; SBP and DBP remain active. Public models have
two outputs because the released experiment targets SBP and DBP.

## Training defaults

- seeds: 20260907, 20260908, 20260909;
- maximum epochs: 60;
- early-stopping patience: 8 epochs;
- batch size: 128 (validation/test use 256 where implemented);
- AdamW learning rate: 0.001;
- weight decay: 0.0001;
- cosine learning-rate schedule;
- gradient-norm clipping: 5.0;
- Huber/Smooth-L1 beta: 1.0 in fold-normalized target units;
- auxiliary HR weight: 0.1;
- face-window augmentation: circular temporal shift of -15 to +15 samples
  (-0.5 to +0.5 s at 30 Hz) plus Gaussian noise with SD 0.02 in z-scored units;
- head dropout: 0.20 for ANN/ResNet/TCN/Transformer and 0.15 for SNN;
- residual/TCN/Transformer internal dropout: 0.10.

Target centering/scaling, session weights, validation selection, early stopping,
and model selection are fitted within the corresponding outer-training fold.
Test performance is not used to choose architecture, threshold, simulation steps,
loss weight, or stopping epoch.

## Aggregation and statistics

Private window predictions are aggregated to one acquisition prediction by the
median. For each seed, absolute errors are averaged within subject and then
equally across subjects. For formal inference, predictions are first averaged
across seeds, then converted to subject errors.

Uncertainty and tests:

- 5,000 subject-cluster bootstrap samples for effect confidence intervals;
- paired two-sided Wilcoxon tests over subject errors;
- Holm family-wise correction over the prespecified leakage family and the
  separately prespecified architecture family;
- between/within-subject variance decomposition and ICC(1,1);
- acquisition-disjoint RBF-SVM identity classification;
- accuracy, balanced accuracy, majority baseline, and 5,000 label permutations.

## Experiment order

1. `run_bspc_extension_experiments.py`: train-median, ANN, and SNN split audit.
2. `run_model_suite_extension.py`: append ResNet-1D, TCN, and Transformer.
3. `revision_audit.py`: overlap, ICC, identity, and agreement analyses.
4. `audit_model_suite_outputs.py`: expected seed/fold/model completeness.
5. figure scripts: use frozen seed metrics and predictions.

## Exploratory boundary

The following analyses were chosen after inspection of baseline results and are
not part of the confirmatory architecture family:

- fingertip PPG teacher and relation/output distillation;
- signal-quality-weighted SNN;
- 32-step SNN, SNN-32-KD, and enhanced SNN-V2;
- broader single-seed classical comparisons.

Report them as exploratory and do not use them to claim that an SNN is superior
to the ANN.
