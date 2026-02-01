# Container image size & CUDA dependency problem

## Current state (2026-02-26)

Build command: `podman build --format docker -t eeg-api -f api.Containerfile .`

Image layers:
- `nvcr.io/nvidia/cuda:12.6.0-cudnn-runtime-ubi9` base — **3.36 GB**
- pip packages added by our Containerfile — **~3.6 GB**
- **Total: ~6.9 GB**

### What's eating the 3.6 GB of pip packages

| Package | Installed size | Notes |
|---|---|---|
| `torch` (`libtorch_cuda.so` + `libtorch_cpu.so`) | **1.8 GB** | unavoidable |
| `triton` (`libtriton.so` + CUPTI backends) | **641 MB** | GPU kernel compiler, not needed for inference |
| `cuda-bindings` (cuda-python) | **106 MB** | CUDA Python API, dep of triton |
| `llvmlite` | 162 MB | needed by numba → librosa → mne-icalabel |
| `scipy`, `pandas`, `mne`, sklearn, etc. | ~450 MB | needed |

---

## The core problem: nvidia-* packages

`torch` from PyPI (and even from the pytorch wheel index) declares 15 `nvidia-*`
packages as pip dependencies. They collectively add ~2 GB in the default install.

We eliminated them by:
1. Installing everything _except_ torch with `pip install -r requirements_other.txt`
2. Installing torch with `pip install --no-deps -r requirements_torch.txt`

This works but requires splitting the requirements into two files via `grep`.

---

## What was tried

### Attempt 1 — `grep` filter in Containerfile (working, but ugly)
Filter `nvidia-*`, `triton`, `cuda-bindings` out of requirements.txt at build
time using grep. Then install torch separately with `--no-deps`.
**Works, produces ~6.9 GB image. User dislikes the grep pattern.**

### Attempt 2 — Poetry `cuda-runtime` optional group + pytorch wheel index source
```toml
[tool.poetry.group.cuda-runtime]
optional = true

[tool.poetry.group.cuda-runtime.dependencies]
triton = "*"
cuda-bindings = "*"
cuda-pathfinder = "*"

[[tool.poetry.source]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
priority = "supplemental"

[tool.poetry.dependencies]
torch = {version = "*", source = "pytorch-cu126"}
```

Result after `poetry lock`:
- `torch = 2.10.0+cu126` ✅ correctly pinned to cu126 build
- `nvidia-*` packages — still 15 entries, all `groups = ["main"]` ❌

**Why it doesn't work:** `nvidia-*` are transitive dependencies of `torch`, which is
a main dependency. Poetry assigns transitive deps to the group of the package that
requires them. Since `torch` is in `main`, all its transitive deps (including
`nvidia-*`) go into `main`. No amount of group configuration can move them out
while `torch` stays in `main`.

`poetry export` has no `--exclude-package` flag. There is **no pure Poetry mechanism**
to exclude transitive dependencies of a main package from the export.

The `cuda-runtime` group **does** cleanly handle `triton` and `cuda-bindings`
because those are declared there directly, not pulled in via `torch`. This is still
useful — it would save ~750 MB without any grep.

---

## Options going forward

### Option A — cuda-runtime group only (partial, saves ~750 MB, no grep)
Keep `cuda-runtime` group in pyproject.toml. Export with `--without cuda-runtime`.
This removes triton + cuda-bindings cleanly. Still install torch with `--no-deps`
to prevent nvidia-* from being pulled back in. Still need to split requirements
into two files (can use grep or a script).

Saves ~750 MB vs current. Image would be ~6.15 GB.

### Option B — Committed requirements files via Makefile (no grep in Containerfile)
Add a `make container-reqs` Makefile target that runs `poetry export`, filters with
grep, and writes committed `requirements-container.txt` + `requirements-torch.txt`.
The Containerfile just does `COPY` + `pip install` — no grep, no logic.
The grep lives in the Makefile (run locally when deps change, result reviewed in git).

### Option C — CPU-only torch for dev, GPU torch in cuda-runtime group
Move `torch` out of main into the `cuda-runtime` group, add a CPU torch to main
for local development from `https://download.pytorch.org/whl/cpu`.
Container exports `--with cuda-runtime --without main-torch` (not directly supported).
Complex to set up, complicates local dev vs container parity.

### Option D — Accept current state
The grep in the Containerfile is a one-liner. It works, it's tested, it's documented.
The image went from 11.3 GB → 6.9 GB. The remaining size is mostly the CUDA base
image (3.36 GB) + torch (1.8 GB), neither of which can be avoided.

---

## What's already committed / changed in this session

- `pyproject.toml`: `python = "~3.12"` (was `^3.12`), pytorch-cu126 source added,
  cuda-runtime group added, torchvision removed (not directly used)
- `poetry.lock`: regenerated for Python ~3.12 + cu126 torch
- `api.Containerfile`: --format docker fix, split requirements approach, LD_LIBRARY_PATH

## Current pyproject.toml state (needs decision)
`torch` is currently pointing at `source = "pytorch-cu126"`. This is fine to keep
(pins the correct CUDA version) but doesn't eliminate nvidia-* deps on its own.
The `cuda-runtime` group is declared but not yet wired into the Containerfile export.

## Recommended next step
Go with **Option B**: add `make container-reqs`, commit the filtered files, simplify
Containerfile to a plain install. Best separation of concerns, grep is hidden in
Makefile, Containerfile stays clean.
