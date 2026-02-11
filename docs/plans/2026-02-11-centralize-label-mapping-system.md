# Centralized Label Mapping System

## Context

The current label mapping system uses magic numbers like `{0: 0, 1: 2}` scattered across multiple files, making it difficult to understand what labels mean and why mappings exist. The user identified this confusion when seeing `{0: 0, 1: 2}` in `prepare_mdd_dataset()` and not understanding its purpose.

**The Problem:**
- Magic numbers require reading comments to understand (what does `2` mean?)
- Label definitions are scattered across 4 different dataset classes
- Dataset-specific remapping logic is hidden in individual `prepare_*_dataset()` functions
- No single source of truth for label conventions
- Hard to verify correctness or extend to new datasets

**Why This Matters:**
When combining datasets in 4-class mode:
- MDD has binary labels: 0=Healthy, 1=Depressed
- CANE/IDUN have 4-class labels: 0=Normal, 1=Anxiety, 2=Depression, 3=Comorbid
- Without remapping, MDD's "depressed" (label 1) would collide with CANE's "anxiety" (label 1)
- The `{0: 0, 1: 2}` mapping fixes this by moving MDD depression to canonical position 2

The solution creates a centralized `thesis/labels.py` module with named constants, Enums, and clear documentation, making the label system self-documenting and maintainable.

## Solution Architecture

### Three-Tier Label System

1. **Raw Labels** (dataset-specific)
   - Each dataset assigns labels when loading files
   - MDDDataset: 0=healthy, 1=depression (binary)
   - CANEDataset/IDUNDataset: 0-3 (4-class)
   - AX_MALIKDataset: all=1 (anxiety only)

2. **Canonical Labels** (4-class convention)
   - 0 = NORMAL (healthy control)
   - 1 = ANXIETY_ONLY
   - 2 = DEPRESSION_ONLY
   - 3 = COMORBID (anxiety + depression)

3. **Binary Labels** (2-class mode)
   - 0 = HEALTHY
   - 1 = PATHOLOGICAL (any disorder)

### Key Design Decisions

1. **Use IntEnum for type safety** - Enums work as integers (backward compatible) but provide names
2. **Centralize in new `thesis/labels.py`** - Single source of truth for all label definitions
3. **Provide mapping functions** - `get_label_mapping_for_dataset()` encapsulates logic
4. **Add validation utilities** - `validate_labels_for_dataset()` catches remapping bugs
5. **Document the WHY** - Each mapping function explains its purpose

## Implementation Steps

### Step 1: Create `thesis/labels.py`

Create new module with:

**A. Canonical Label Enums**
```python
class CanonicalLabel(IntEnum):
    NORMAL = 0
    ANXIETY_ONLY = 1
    DEPRESSION_ONLY = 2
    COMORBID = 3

class BinaryLabel(IntEnum):
    HEALTHY = 0
    PATHOLOGICAL = 1
```

**B. Dataset-Specific Raw Label Enums**
```python
class MDDRawLabel(IntEnum):
    HEALTHY = 0
    DEPRESSION = 1

class CANERawLabel(IntEnum):
    NORMAL = 0
    ANXIETY = 1
    DEPRESSION = 2
    COMORBID = 3

class IDUNRawLabel(IntEnum):
    NORMAL = 0
    ANXIETY = 1
    DEPRESSION = 2
    COMORBID = 3

class AXMALIKRawLabel(IntEnum):
    ANXIETY = 1
```

**C. Mapping Functions**

```python
def get_mdd_to_canonical_mapping() -> Dict[int, int]:
   """
   Map MDD binary labels to canonical 4-class labels.

   MDD: 0=healthy, 1=depression
   Canonical: 0=normal, 2=depression

   This prevents MDD depression (1) from being mislabeled as anxiety (1).
   """
   return {
      MDDRawLabel.HEALTHY: CanonicalLabel.HEALTHY,
      MDDRawLabel.DEPRESSION: CanonicalLabel.DEPRESSION_ONLY,
   }


def get_canonical_to_binary_mapping() -> Dict[int, int]:
   """Collapse 4-class to binary (all pathological → 1)."""
   return {
      CanonicalLabel.HEALTHY: BinaryLabel.HEALTHY,
      CanonicalLabel.ANXIETY_ONLY: BinaryLabel.PATHOLOGICAL,
      CanonicalLabel.DEPRESSION_ONLY: BinaryLabel.PATHOLOGICAL,
      CanonicalLabel.COMORBID: BinaryLabel.PATHOLOGICAL,
   }


def get_label_mapping_for_dataset(
        dataset_name: str,
        num_classes: int,
        explicit_mapping: Optional[Dict[int, int]] = None,
) -> Optional[Dict[int, int]]:
   """
   Get appropriate label mapping for a dataset and classification mode.

   Priority:
   1. explicit_mapping (if provided)
   2. Binary mode: apply canonical → binary collapse
   3. 4-class mode: apply dataset-specific remapping (MDD only)
   """
   if explicit_mapping is not None:
      return explicit_mapping

   if num_classes == 2:
      return get_canonical_to_binary_mapping()

   if num_classes == 4 and dataset_name == "mdd":
      return get_mdd_to_canonical_mapping()

   return None  # CANE/IDUN/AX_MALIK use canonical labels as-is
```

**D. Validation & Utility Functions**
```python
def get_canonical_label_name(label: int) -> str:
    """Get human-readable name for canonical label."""
    try:
        return CanonicalLabel(label).name.replace("_", " ").title()
    except ValueError:
        return f"Unknown({label})"

def validate_labels_for_dataset(
    labels: set[int],
    dataset_name: str,
    num_classes: int,
) -> None:
    """
    Validate labels match expected values for dataset.

    Raises ValueError if:
    - MDD in 4-class has anxiety (1) or comorbid (3) labels
    - Binary mode has labels outside {0, 1}
    - Unexpected label combinations
    """
    # Implementation validates per-dataset constraints
```

### Step 2: Update `thesis/data_preparation.py`

**Replace `determine_num_classes()` function (lines 37-53):**

```python
# OLD:
def determine_num_classes(class_mode: str) -> tuple[int, Optional[dict[int, int]]]:
    if class_mode == "2":
        return 2, {0: 0, 1: 1, 2: 1, 3: 1}  # Magic numbers!
    elif class_mode == "4":
        return 4, None

# NEW:
from thesis.labels import get_canonical_to_binary_mapping

def determine_num_classes(class_mode: str) -> tuple[int, Optional[dict[int, int]]]:
    """
    Determine number of classes and label mapping based on class_mode.

    Note: Returns binary collapse mapping for 2-class mode.
    Dataset-specific remapping (e.g., MDD) is handled in prepare_*_dataset().
    """
    if class_mode == "2":
        return 2, get_canonical_to_binary_mapping()
    elif class_mode == "4":
        return 4, None
    else:
        raise ValueError(f"Invalid class_mode: {class_mode}")
```

**Simplify `prepare_mdd_dataset()` (lines 147-154):**

```python
# OLD:
effective_label_mapping = label_mapping
if label_mapping is None and num_classes == 4:
    effective_label_mapping = {0: 0, 1: 2}  # Confusing!
    logger.info(f"Applying MDD label remapping: {effective_label_mapping}")

# NEW:
from thesis.labels import get_label_mapping_for_dataset

effective_label_mapping = label_mapping
if label_mapping is None and num_classes == 4:
    effective_label_mapping = get_label_mapping_for_dataset("mdd", num_classes)
    if effective_label_mapping:
        logger.info(
            f"Applying MDD label remapping for 4-class mode: {effective_label_mapping} "
            f"(maps MDD depression=1 to canonical DEPRESSION_ONLY=2)"
        )
```

### Step 3: Update Dataset Classes (Optional Enhancement)

In `thesis/dataset.py`, update raw label assignments to use enums for clarity:

**CANEDataset (line 342):**

```python
# OLD:
LABEL_INT_MAP = {"normal": 0, "anxiety": 1, "depression": 2, "anxiety-depression": 3}

# NEW (optional, for clarity):
from thesis.labels import CANERawLabel

LABEL_INT_MAP = {
   "normal": int(CANERawLabel.HEALTHY),
   "anxiety": int(CANERawLabel.ANXIETY),
   "depression": int(CANERawLabel.DEPRESSION),
   "anxiety-depression": int(CANERawLabel.COMORBID),
}
```

**Note:** This step is optional - the existing integer assignments work fine. The primary benefit is documentation.

### Step 4: Update Tests

Enhance `tests/test_label_remapping.py` with validation:

```python
from thesis.labels import (
   validate_labels_for_dataset,
   CanonicalLabel,
   BinaryLabel,
   get_mdd_to_canonical_mapping,
)


def test_mdd_4class_mode_labels(rng):
   """Test MDD in 4-class mode uses only {0, 2} labels."""
   flat_dataset, _ = prepare_mdd_dataset(
      conditions=["EC"],
      channel="Fp1",
      rng=rng,
      num_classes=4,
      test_mode=False,
   )

   labels = {flat_dataset[i][1] for i in range(len(flat_dataset))}

   # Use centralized validation
   validate_labels_for_dataset(labels, "mdd", num_classes=4)

   # Self-documenting assertions
   assert labels == {CanonicalLabel.HEALTHY, CanonicalLabel.DEPRESSION_ONLY}
   assert CanonicalLabel.ANXIETY_ONLY not in labels
   assert CanonicalLabel.COMORBID not in labels


def test_label_mapping_functions():
   """Test centralized mapping functions."""
   mdd_mapping = get_mdd_to_canonical_mapping()
   assert mdd_mapping == {0: 0, 1: 2}  # Still verify correct values

   # But now it's self-documenting
   assert mdd_mapping[MDDRawLabel.HEALTHY] == CanonicalLabel.HEALTHY
   assert mdd_mapping[MDDRawLabel.DEPRESSION] == CanonicalLabel.DEPRESSION_ONLY
```

### Step 5: Update Documentation

Update `CLAUDE.md` section on labels:

```markdown
## Label System

All label definitions are centralized in `thesis/labels.py`. This module provides:

- **Canonical 4-class labels**: Used when combining datasets
  - 0 = NORMAL (healthy control)
  - 1 = ANXIETY_ONLY
  - 2 = DEPRESSION_ONLY
  - 3 = COMORBID (both)

- **Binary labels**: Used for 2-class classification
  - 0 = HEALTHY
  - 1 = PATHOLOGICAL (any disorder)

- **Dataset-specific raw labels**: What each dataset assigns initially
  - MDDDataset: 0=healthy, 1=depression (binary convention)
  - CANEDataset/IDUNDataset: 0-3 (4-class convention)
  - AX_MALIKDataset: all subjects get 1=anxiety

- **Mapping functions**:
  - `get_label_mapping_for_dataset(dataset_name, num_classes)` - get correct mapping
  - `get_mdd_to_canonical_mapping()` - MDD binary → canonical 4-class
  - `get_canonical_to_binary_mapping()` - canonical 4-class → binary

**Key Insight:** MDD uses binary labels (0=healthy, 1=depression). In 4-class mode,
MDD depression must map to canonical label 2 (DEPRESSION_ONLY), not 1 (which is ANXIETY_ONLY).
This prevents confusion when combining MDD with CANE/IDUN datasets.

See `thesis/labels.py` for detailed documentation and usage examples.
```

## Critical Files

- **NEW: `/home/milan/Documents/diplomka/code/thesis/labels.py`** - Centralized label definitions
- `/home/milan/Documents/diplomka/code/thesis/data_preparation.py` - Update `determine_num_classes()` and `prepare_mdd_dataset()`
- `/home/milan/Documents/diplomka/code/tests/test_label_remapping.py` - Add validation tests
- `/home/milan/Documents/diplomka/code/CLAUDE.md` - Update label system documentation
- (Optional) `/home/milan/Documents/diplomka/code/thesis/dataset.py` - Use enums for clarity

## Verification Steps

### 1. Run Existing Tests
```bash
poetry run pytest tests/test_label_remapping.py -v
```
All existing tests should pass without modification (backward compatible).

### 2. Test Label Validation
```bash
poetry run pytest tests/test_label_remapping.py::test_label_mapping_functions -v
poetry run pytest tests/test_label_remapping.py::test_mdd_4class_mode_labels -v
```

### 3. Manual Verification
```python
from thesis.labels import (
    get_label_mapping_for_dataset,
    CanonicalLabel,
    get_mdd_to_canonical_mapping,
)

# Verify MDD mapping
mdd_mapping = get_label_mapping_for_dataset("mdd", num_classes=4)
print(mdd_mapping)  # Should be {0: 0, 1: 2}

# Verify enum values
print(CanonicalLabel.DEPRESSION_ONLY)  # Should be 2
print(CanonicalLabel.ANXIETY_ONLY)     # Should be 1

# Verify self-documenting
assert mdd_mapping[MDDRawLabel.DEPRESSION] == CanonicalLabel.DEPRESSION_ONLY
```

### 4. Integration Test
Run a small training experiment to ensure end-to-end functionality:
```bash
poetry run python main.py train \
  --dataset all \
  --class-mode 4 \
  --channel Fp1 \
  --condition ec \
  --test-mode \
  --checkpoint-dir experiments/test_labels
```

Check logs for:
- Correct label remapping messages
- No label mismatch errors
- Confusion matrix shows labels 0, 1, 2, 3 (not collisions)

### 5. Code Quality
```bash
poetry run ruff check thesis/labels.py
poetry run mypy thesis/labels.py
```

## Expected Outcomes

**Before:**
- `{0: 0, 1: 2}` ← What does this mean?
- Label definitions scattered across 4 files
- Hard to verify correctness

**After:**
- `get_mdd_to_canonical_mapping()` ← Self-explanatory function name
- All labels defined in `thesis/labels.py`
- Easy to understand: `MDDRawLabel.DEPRESSION` → `CanonicalLabel.DEPRESSION_ONLY`
- Validation catches bugs: `validate_labels_for_dataset()` ensures MDD never has anxiety labels

## Migration Notes

- **Backward Compatible**: IntEnum values work as integers, existing code unchanged
- **Incremental Adoption**: Can update dataset classes to use enums gradually
- **No Breaking Changes**: Same mappings, just centralized and documented
- **Test Coverage**: Existing tests verify behavior unchanged
