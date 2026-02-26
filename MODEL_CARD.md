# Dataset Overview

> **CANE and IDUN are paired recordings from the same subjects** — cap EEG (CANE) and
> in-ear EEG (IDUN) were collected simultaneously in the same study.
> When `--channel in-ear` is used, CANE is automatically replaced by IDUN.

---

## Technical Specifications

| Property              | MDD                              | CANE                             | IDUN                     | SAD                              |
|-----------------------|----------------------------------|----------------------------------|--------------------------|----------------------------------|
| Target condition      | Major Depression                 | Anxiety / Dep. / Comorbid        | ← same as CANE           | Social Anxiety                   |
| Paper                 | Mumtaz et al. 2017               | —                                | —                        | Al-Ezzi et al. 2023              |
| Device                | Brain Master Systems             | —                                | IDUN 'Guardian' (in-ear) | egosports + ANT Neuro cap        |
| Sampling rate         | 256 Hz                           | 500 Hz                           | 250 Hz                   | 2048 → 256 Hz                    |
| Electrodes (total)    | 19-ch cap                        | 8-ch cap                         | 1 (in-ear)               | 30-ch cap                        |
| Channels used         | 8 scalp (10–20)                  | 8 scalp (10–20)                  | 1 in-ear                 | 8 scalp (10–20)                  |
| Channels used (names) | Fp1, Fp2, C3, Cz, C4, T7, T8, O2 | Fp1, Fp2, T7, C3, Cz, C4, T8, Oz | in-ear                   | Fp1, Fp2, C3, Cz, C4, T7, T8, O2 |
| Reference             | Linked-ear → infinity (IR)       | —                                | —                        | CPz (ground: AFz)                |
| File format           | EDF                              | CSV                              | CSV + quality file       | EDF                              |
| Recording paradigm    | Resting state                    | Resting state                    | Resting state            | Resting state                    |
| Duration              | 5 min EC + 5 min EO              | —                                | —                        | ~4–6 min (~120 s used)           |
| Diagnosis tool        | DSM-IV · BDI-II · HADS           | —                                | —                        | SIAS                             |

---

## Subject & Chunk Counts

| Class                | MDD                | CANE        | IDUN        | SAD      |
|----------------------|--------------------|-------------|-------------|----------|
| Healthy / Normals    | H: 28              | Normals: 20 | Normals: 17 | HC: 22   |
| Anxiety              | —                  | AX: 19      | AX: 20      | SAD: 22  |
| Depression           | MDD: 30 EC / 32 EO | DEP: 2      | DEP: 1      | —        |
| Comorbid             | —                  | COM: 27     | COM: 15     | —        |
| **Total files EC**   | **58**             | **68**      | **53**      | **44**   |
| **Total files EO**   | **60**             | **68**      | **53**      | **44**   |
| **Chunks EC (10 s)** | **1 741**          | **~1 972**  | **~1 352**  | **~528** |
| **Chunks EO (10 s)** | **1 794**          | **~1 929**  | **~1 304**  | **~528** |

> IDUN chunk counts are after quality filtering (chunks with quality = 0 rejected).
> MDD: paper reports 33 MDD + 30 HC; 5 subjects absent from available files.

---

## References

**MDD** — Mumtaz W. et al. "Electroencephalogram (EEG)-based computer-aided technique to diagnose
major depressive disorder (MDD)." *Biomedical Signal Processing and Control* 31 (2017) 108–115.
DOI: 10.1016/j.bspc.2016.07.006

**SAD** — Al-Ezzi A. et al. "Machine Learning for the Detection of Social Anxiety Disorder Using
Effective Connectivity and Graph Theory Measures." *IEEE Trans. Neural and Rehab. Eng.* (2023).

**Surrogate in-ear channel creation** — Tremmel C. et al. "Estimating cognitive workload using a commercial in-ear EEG
headset." *Journal of Neural Engineering* 21 (2024) 066022. DOI: 10.1088/1741-2552/ad8ef8
*(cited as methodological basis for in-ear EEG; the clinical CANE/IDUN data is from a separate study)*