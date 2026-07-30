### MDD Dataset

Expected location at `../MDD/`. Files follow the naming pattern: `{H|MDD} S{N} {EC|EO|TASK}.edf`

**Dataset Structure:**
- **H** = Healthy control subjects
- **MDD** = Major Depressive Disorder subjects
- **Conditions**: EC (Eyes Closed), EO (Eyes Open), TASK - can train on single or combined (ec+eo)
- **Channels**: 8 channels used (Fp1, Fp2, T7, T8, C3, C4, Cz, Oz) with standardized 10-20 naming, plus synthetic in-ear option
- **Sampling Rate**: 256 Hz native (verified against all 181 EDF headers), resampled to 250 Hz before the STFT
- **Segments**: 10-second chunks (2,560 samples each at the native rate)

### CANE Dataset

Expected location at `../CANE/`. Files follow pattern: `{H|AX} S{N} {ec|eo}.edf`

**Dataset Structure:**
- **H** = Healthy control subjects
- **AX** = Anxiety disorder subjects
- **Conditions**: ec (eyes closed), eo (eyes open) - can train on single or combined (ec+eo)
- **Channels**: Same 8 channels as MDD dataset with standardized naming (T3→T7, T4→T8), plus synthetic in-ear option
- **Sampling Rate**: 500 Hz
- **Preprocessing**: Optional artifact removal via `--skip-artifact-removal` flag

**Note:** When using `--channel in-ear`, CANE is automatically replaced with the IDUN dataset (see below) for real in-ear EEG recordings.

### IDUN Dataset

Expected location at `../IDUN_IN_EAR/`. Files follow pattern: `{class_dir}/{subject_id}/eeg_{subject_id}{condition}.csv`

**Dataset Structure:**
- **Real in-ear EEG recordings** from IDUN device (single-channel CSV format)
- **Classes**: normals (17), anxiety (20), depression (1), comorbid (15) - 53 usable subjects
- **Conditions**: ec (eyes closed), eo (eyes open) - case-insensitive
- **Channel**: Single in-ear channel (no multi-channel support)
- **Sampling Rate**: 250 Hz (verified from timestamps)
- **Quality Metrics**: Signal quality data available in `quality_*.csv` files (0=not measured, 90+=good)
- **Preprocessing**: z-score → detrend → bandpass 1-70 Hz → notch 50 Hz → 10s chunking → quality-based rejection

**Usage:**
- IDUN is **automatically used** when `--channel in-ear` is specified with `--dataset cane` or `--dataset all`
- Provides real in-ear EEG instead of synthetic bipolar derivations (T8-T7)
- Occupies the "CANE" slot in fold creation, so logs may report "CANE" but IDUN data is actually used
- Quality threshold (default 0.0) rejects chunks with unmeasured signal quality

### SAD Dataset

Expected location at `../SAD/`. Files follow pattern: `{ec|eo}/C{N}.edf`

**Dataset Structure:**
- **All subjects are anxiety class** (no healthy controls in this dataset)
- **Conditions**: EC (eyes closed), EO (eyes open) - can train on single or combined (ec+eo)
- **Channels**: Same 8 channels as MDD dataset (Fp1, Fp2, C3, Cz, C4, T7, T8, O2), plus synthetic in-ear option
- **Sampling Rate**: 256 Hz
- **Duration**: 120 seconds per file
- **Subjects**: 21 subjects (42 files total - one EC and one EO per subject)
- **Preprocessing**: Uses MDDDataset preprocessing pipeline via inheritance
