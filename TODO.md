Implement the plan: docs/plans/2026-02-01-ax-malik-dataset-design.md

batch normalization?

Things to think about: 

Q: how long recordings do I need for X pct validations? - plot some graph in evaluation
- [ ] My preprocessing is still using all channels so the question is if to do the preprocessing on chunk level to get
  closer to online and to one channel or to do multichannel preprocessing - i don't think I'm getting a lot of benefit
  from the full channel preprocessing

What can you tune:

- learning rate, batch size, dropout
- **is it worth adding augmentation, what types on EEG data?** (performance should stay similar for robust model)
- **pretraining/using all channels**
- **making the model smaller?**
- k in k-folds
- eyes open vs eyes closed
- they don't say how they work with multichannel

> Where did I get this from? batch size 128 (probably individual chunks), adam, cross entropy, learning rate 0.0001

- [ ] Check what [Claude has suggested](https://claude.ai/share/8b2c46f9-19d3-411f-acac-ae3c87e180ba) to fight overfitting=
- [ ] Explain to me why are there frequencies >70Hz when it should be fitlered
- [ ] Also, I hope the tensor you provide to the network doesn't contain these frequencies, would be a waste
- [ ] What is the frequency range of the spectogram? If you cropped it at 70Hz, maybe that's all you should provide
- [ ] CANE the spectogram goes into higher frequencies then it should -> poor filtering
- [ ] Review all TODOs and resolve them
- [ ] Manually remove files and train one epoch
- [ ] Move datasets into code repo

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


from Depcap

34 MDD, 30 healthly, 10-20 system: Fp1, Fp2, F3, F4, F7, F8, Fz, T3, T4,
T5, T6, P3, P4, Pz, O1, O2, C3, C4, and Cz.

ale v tom co jsem stahnul je jich min nebo co

---


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


Evaluation logic

- Eyes closed only
- one channel only
- 10 fold cross validation 
