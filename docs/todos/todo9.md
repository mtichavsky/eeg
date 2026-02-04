# Task: Add DASPS Dataset Dataloader

## Objective
Implement a new dataset class for the DASPS dataset following the existing architecture patterns in `thesis/dataset.py` (similar to `MDDDataset`, `CANEDataset`, `AXMalikDataset`).

## Implementation Steps

1. **Understand DASPS preprocessing:**
   - Read `DASPS/README_PREPROCESSING.md` to understand the preprocessing pipeline
   - Explore the DASPS directory structure to identify preprocessed files (PREPROCESS_MAT/)
   - Summarize what preprocessing was already performed (e.g., filtering, artifact removal, segmentation) reading up
     the documentation in the directory

2. **Create DASPSDataset class:**
   - Implement in `thesis/dataset.py`
   - Follow existing patterns: lazy/preload modes, LRU caching, PyTorch Dataset interface
   - Use the same 8-channel canonical ordering: Fp1, Fp2, T7, T8, C3, C4, Cz, Oz
   - Map any non-standard channel names to 10-20 system (similar to T3→T7, T4→T8 mapping)

3. **Handle preprocessing:**
   - Load preprocessed files (don't re-preprocess for now)
   - Ensure 10-second chunks (2500 samples for consistency)
   - Match the preprocessing pipeline structure used by other datasets

4. **Implement labeling strategy:**
   - **DECISION NEEDED:** How to handle mild depression?
     - Option A: Treat mild depression as healthy (0), moderate/severe as depression (1)
     - Option B: Exclude mild depression subjects entirely from dataset
   - Label mapping for binary classification:
     - 0 = Healthy (including mild depression if Option A)
     - 1 = Depression (moderate/severe)

5. **Add dataset preparation function:**
   - Create `prepare_dasps_dataset()` function following the pattern of `prepare_mdd_dataset()` and `prepare_cane_dataset()`
   - Handle file path construction, label extraction, and subject ID parsing
   - Support eyes closed (EC) / eyes open (EO) / combined conditions

6. **Integration:**
   - Add DASPS to `--dataset` CLI options in `thesis/cli.py`
   - Support `--dataset dasps` for standalone use
   - Support `--dataset all` to include DASPS with MDD/CANE/AX_MALIK in multi-dataset training
   - Update `create_balanced_folds()` if needed to handle DASPS in combined datasets

## Questions to Answer
- What is the exact directory structure and file naming pattern in DASPS?
- What preprocessing has already been performed (filters, sampling rate, etc.)?
- What are the severity levels available in DASPS labels?
- How many subjects and files are in the dataset?
- Should mild depression be treated as healthy or excluded?

## Expected Output
- New `DASPSDataset` class in `thesis/dataset.py`
- New `prepare_dasps_dataset()` function
- Updated CLI to support `--dataset dasps`
- Documentation of labeling decisions and dataset characteristics