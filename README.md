Ve vlaku:
- [ ] precist ten paper
- [ ] precist Markov - jak resi tyto problemy

- [ ] Pridej ten run subcommand, make sure ze to funguje tak jak ma - tohle je asi prvni vec co chci udelat
- [ ] checkpoints - pokud ta slozka neco obsahuje, tak nespoustej trenovani
- [ ] still missing 2 channels, find the one which is closest to the in-ear

Dataset refactor:
- ✔ dependency injection -> dataset into spectogram dataset
- og dataset - no preload, no caching
- nad nim budu mit udelane splity abych mohl delat 10fold
- doufam, ze se mi podari odstranit tohle delani tech chunku, to je uplne napicu, viz logy nize
- jak budu mit ten druhy dataset, tak pro nej jenom udelam separatni tridu a tu pak jebnu stejnym zpusobem do toho 
  spectogram datasetu
- ✔ ten preprocessing je specificky pro dataset i model, ale klidne to bake-in do toho datasetu
- ✔ make it possible to pick one of those channels
- ideally supporting lazy loading in that spectogram dataset
- maybe transform the data only through transform function - to_spectogram and to_wavelet, using lambda you can combine them
- but it still has to support lazy loading
- maybe I don't need the second dataset abstraction
- you can even implement it throught that lsu cache mechanism - if it's 100 file entries, it will be fine
- they might be using 0.6 dropout
- batch size 128 (probably individual chunks), adam, cross entropy, learning rate 0.0001


```
[2025-10-31 16:56:25,378 INFO dataset.collate_variable_length_eeg] Trimming batch chunks to 29 chunks.
[2025-10-31 16:56:29,522 INFO dataset.collate_variable_length_eeg] Trimming batch chunks to 18 chunks.
```

- [ ] Combining multiple channels - vubec nevim jak to delaji
- [ ] zaroven velikost toho jejich spektogramu je vetsi -> my second convolution is 5x5, not 15x15
- 
> With 10-second segments, you can't get (254, 342) dimensions using standard STFT. Even with maximum overlap:
> - nperseg=250 → 126 frequency bins (not 254)

> In this work, an automated depression detection system Dep-
> Cap is proposed wherein pre-processed 1D EEG is first
> converted into 2D spectrogram images using STFT. Here,
> a medium size Kaiser window of size 2n − 1 is used for
> windowing, where n is the number of bits.

- 10 seconds → ~21-80 time bins depending on overlap (not 342)
- [ ] Spectogram window - pouzivaji asi neco jineho nez ja

STFT:
In this work, a hamming window of varied window size
between 0.2 to 1 second is used for achieving optimum
results

pre-processed 1D EEG is first
converted into 2D spectrogram images using STFT. Here,
a medium size Kaiser window of size 2n − 1 is used for
windowing, where n is the number of bits.

specificity and this shit they show equations in the paper

The reliability and robustness of the proposed CNN-LSTM
model are also examined by adding white noises of different
SNRs ( 0, 10, 20, 30, 40, and 50 dB). It can be seen from
---


Evaluation logic

- kazdy 10sekundovy chunk vyhodnocuju zvlast, na konec agreguju pro danou nahravku vsechny chunky, abych dostal vysledek
- Eyes closed only
- one channel only
- 10 fold cross validation 

MDD dataset

from depcap

34 MDD, 30 healthly, 10-20 system: Fp1, Fp2, F3, F4, F7, F8, Fz, T3, T4,
T5, T6, P3, P4, Pz, O1, O2, C3, C4, and Cz.

Participant’s EEG was recorded for 10 minutes, comprising
5 minutes with their Eyes Open (EO) and 5 minutes with Eyes
Closed (EC).

this is citation
choppy activities in depressed EEG. A Butterworth band pass
filter of cut-off frequency 0.5Hz-70Hz and a notch filter of
50Hz is used to remove the power grid effect
Independent Component Analysis
(ICA) is used to reduce artifacts caused by patient movement
and eye blinking. Then, the whole EEG signal of 5 min
duration is segmented into the interval of 10s. Finally, the Z
score normalization technique is used for amplitude scaling
of each EEG segment before being sent to the proposed neural
network.


sth to put into your thesis
STFT is
an extension of simple Fourier transform, that leads to better
spectral analysis to extract features and detect abnormalities
in EEG signals at every instant. STFT is
an extension of simple Fourier transform, that leads to better
spectral analysis to extract features and detect abnormalities
in EEG signals at every instant
These networks are preferred over
traditional machine learning methods for several reasons:
1) CNNs can learn a high-level representation of data directly
from the input, unlike other machine learning algorithms
requiring hand-crafted features. 2) CNNs use convolutional
filters, which are very efficient in extracting spatial infor-
mation which other machine-learning models can miss.
3) CNNs can be trained more efficiently on larger datasets
than other machine learning models. 4) CNNs can be opti-
mized for speed and complexity by altering various parame-
ters and fortifying the network’s architecture.
