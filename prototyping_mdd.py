from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from mne_icalabel import label_components
from scipy.signal import stft
from scipy.stats import zscore

MDD_DIR = Path("../MDD/")

raw = mne.io.read_raw_edf(MDD_DIR / "MDD S1 EC.edf", preload=True)
print(raw.info)

_ = raw.plot()
plt.show()
