```bash
python main.py run ../checkpoints/fold_1_best.pth ../MDD/MDD\ S26\ EC.edf --channel Fp1 --skip-ica
python main.py run checkpoints/fold_1_best.pth ../MDD/MDD\ S26\ EC.edf --channel Fp1 --skip-ica
```

2. Condition-specific metrics calculation (main.py:346-356): The substring matching approach "EC" in subj_id works but is fragile - if
   subject IDs ever change format, this breaks silently

write some evaluation framework for yourself


- Rationale: Chunk-level accuracy provides more stable gradient signal (more samples)
- Trade-off: Subject-level accuracy is what matters clinically, but has higher variance
- Risk: A model could overfit to chunks while subject accuracy plateaus or decreases

Recommendation: This change is good, but consider logging both metrics and alerting if they diverge significantly (e.g., chunk acc
increasing while subject acc decreasing = overfitting to chunk noise).

- play with different learning rate and batch size, maybe smaller batch and lr would make it more stable
- I cannot train both at T7 right?
- you can do augmentation if you want right?
- pretraining on all channels, somehow using all channels to get more data? 
- use smaller k-fold - maybe the eval will be better, but less training data
- Maybe make sure I'm trainig only on chunk metric, not on the combined, WHEN IT COMES TO ACCURACY
- if you add eyes open, it's the same as augumentation basically, try it with it too

Held-out Test Set - Critical with small data - keep a completely separate test set that you never tune on. Final performance here is your ground truth
Data Augmentation Test - If you can augment your data (noise, transformations), does performance stay stable? Good models are robust; overfit models are fragile.

- [ ] Check what [Claude has suggested](https://claude.ai/share/8b2c46f9-19d3-411f-acac-ae3c87e180ba) to fight overfitting=
- [ ] Explain to me why are there frequencies >70Hz when it should be fitlered
- [ ] Also, I hope the tensor you provide to the network doesn't contain these frequencies, would be a waste
- [ ] What is the frequency range of the spectogram? If you cropped it at 70Hz, maybe that's all you should provide
- [ ] Unit/automatic testing of what I have written - what do you test in these ML applications - is it more like you
  see the results of the model /how good it is and that ´s it ? Bcs thats kinda the validation ? Or do you write unit
  tests?
- [ ] Review all TODOs and resolve them
- [ ] check that the chunking logic works properly via some prints, check lazy loading works properly
- [ ] finish localizing those channels, so that I have the proper ones
- How did they work with multi channel? Did they use both eyes open/eyes closed? they don't say
- choose the right channel when working with one only
- [ ] manually remove files and train one epoch
- [ ] how long recordings do I need for X pct validations? - plot some graph in evaluation
- [ ] maybe add some augumentations
- [ ] My preprocessing is still using all channels so the question is if to do the preprocessing on chunk level to get
      closer to online and to one channel or to do multichannel preprocessing
- [ ] CANE the spectogram goes into higher frequencies then it should -> poor filtering

CANE:

Things I Skipped That You Could Add Later:
- ICA (Independent Component Analysis)
    - Would require converting to MNE format
    - Most effective but computationally expensive 
    - -Could add as optional like in MDD pipeline
- Wavelet denoising
  - Could be more effective than simple filtering
  - Preserves transients better
- ASR (Artifact Subspace Reconstruction)
  - State-of-the-art for continuous artifact removal
  - Would need additional libraries
- Adaptive filtering
  - Could better handle time-varying noise
- Bad channel detection and interpolation
  - The multi-channel function starts this but could be more sophisticated
- EOG/EMG regression
  - If you have reference channels
- More sophisticated resampling
  - Anti-aliasing filter before downsampling

---

Loss function design:
`combined_metric = chunk_metrics["accuracy"] * subject_metrics["accuracy"]`
This is probably the best right? But do I have it baked into training? No.
So I need to clear up this inconsistency, cause training runs surely on chunk_metrics only
if it was possible to calculate loss like that 
Right now, training is done based on chunk accuracy, best model based on combine metrics eval results.
That's kinda inconsistent



from depcap

34 MDD, 30 healthly, 10-20 system: Fp1, Fp2, F3, F4, F7, F8, Fz, T3, T4,
T5, T6, P3, P4, Pz, O1, O2, C3, C4, and Cz.

ale v tom co jsem stahnul je jich min nebo co

---

Dataset refactor:
- og dataset - no preload, no caching - double check 
- jak budu mit ten druhy dataset, tak pro nej jenom udelam separatni tridu a tu pak jebnu stejnym zpusobem do toho 
  spectogram datasetu
- ideally supporting lazy loading in that spectogram dataset
- maybe transform the data only through transform function - to_spectogram and to_wavelet, using lambda you can combine them
- but it still has to support lazy loading
- they might be using 0.6 dropout
- batch size 128 (probably individual chunks), adam, cross entropy, learning rate 0.0001

- [ ] Combining multiple channels - vubec nevim jak to delaji - tohle muzu udelat podle sebe asi
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
