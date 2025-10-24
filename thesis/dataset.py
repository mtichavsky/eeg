from mne.preprocessing import ICA
from mne_icalabel import label_components
from pathlib import Path
import mne

CANE_DIR = Path("../CANE/")
MDD_DIR = Path("/home/milan/Documents/diplomka/MDD/")

SEGMENT_LENGTH = 10
SFREQ = 1000 / 4
CHUNK_SAMPLES = int(SEGMENT_LENGTH * SFREQ)


def get_preprocessed_chunks():
    raw = mne.io.read_raw_edf(MDD_DIR / "MDD S1 EC.edf", preload=True)
    raw = raw.filter(l_freq=1, h_freq=70, method="iir")
    raw = raw.notch_filter(freqs=50)

    # TODO not sure about Cz; Add T7, T8 - ma to nejake ackove, chceckni ten clanek
    raw = raw.pick(["EEG Fp1-LE", "EEG Fp2-LE", "EEG C3-LE", "EEG C4-LE", "EEG O2-LE", "EEG Cz-LE"])
    mapping = {
        "EEG Fp1-LE": "Fp1",
        "EEG Fp2-LE": "Fp2",
        "EEG C3-LE": "C3",
        "EEG C4-LE": "C4",
        "EEG O2-LE": "O2",
        "EEG Cz-LE": "Cz",
    }
    raw = raw.rename_channels(mapping)

    filt_raw = raw.set_eeg_reference("average")
    ica = ICA(
        max_iter="auto",
        method="infomax",
        random_state=97,  # seed?
        fit_params=dict(extended=True),
    )
    ica.fit(filt_raw)

    montage = mne.channels.make_standard_montage("standard_1020")
    # 2. Apply montage (MNE will keep only those channels that exist in raw)
    filt_raw = filt_raw.set_montage(montage, match_case=False)
    ic_labels = label_components(filt_raw, ica, method="iclabel")

    labels = ic_labels["labels"]
    print(labels)
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    print(f"Excluding these ICA components: {exclude_idx}")
    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    preprocessed = filt_raw.copy()
    ica.apply(preprocessed, exclude=exclude_idx)

    df = preprocessed.to_data_frame()
    chunks = [df.iloc[i : i + CHUNK_SAMPLES] for i in range(0, len(df), CHUNK_SAMPLES)]
    return chunks
