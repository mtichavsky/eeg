from pathlib import Path

import matplotlib.pyplot as plt
import mne

MDD_DIR = Path("../MDD/")

raw = mne.io.read_raw_edf(MDD_DIR / "MDD S1 EC.edf", preload=True)
print(raw.info)

_ = raw.plot()
plt.show()
