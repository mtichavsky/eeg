projekt
- zadani
- chce kod/ciste report?
- potrebuju pak slides
- jak moc se zamerit na jeden channel x vsechny channely

nechce slidy, staci mu report, vicemene jak chapeme ty metody z prednasek

- jak poznam ze model je dobry - projit modely
  - jak moc je to spatne a jak moc je to proste EEG data problem
- all channel x single channel
  - mam delat 3D konvoluce x nezavisle modely - oni to tam nezminujou, jak velky je model v tom clanku, abych ja nemel
    moc velky model na signle channel, treba ten jejich model zvladal vsechny channels
  - oni tam nepisou nic o cross fold
- online x offline (ICA, common noise)
  - user raw, focus on one channel
> `combined_metric = chunk_metrics["accuracy"] * subject_metrics["accuracy"]`
- we kinda skipped this one
- early stopping based on accuracy
- mam delat augmentace, I think they were adding noise
- kombinovat eyes closed x eyes open - deep learning - i can try both, otherwise they treat it separate
- co mam mit v lednu hotove?

chunks sizes - 1-10s 
write 4 point into report, make sure to submit by the end of the year

30 Jan 10:40 am

- [ ] Convert to chunk accuracy
- 
Evaluation:

- make sure imbalanced classes don't cause you any problems
- make sure you're not overfitting the model
- early stopping based on chunk level, loss based on chunk level 
  - Risk: A model could overfit to chunks while subject accuracy plateaus or decreases
  - **TODO:** log both metrics and compare, ideally show them on the graphs

> Commentary : Loss function design:
> `combined_metric = chunk_metrics["accuracy"] * subject_metrics["accuracy"]`
> This is probably the best right? But do I have it baked into training? No.
> So I need to clear up this inconsistency, cause training runs surely on chunk_metrics only
> if it was possible to calculate loss like that
> Right now, training is done based on chunk accuracy, best model based on combine metrics eval results.
> That's kinda inconsistent
> Come up with ideas how to make sure :)

- batch normalization?
- vanishing gradient problem?

**TODO** train_one_fold - clear up the chunks vs combined accuracy problem

## What to evaluate on:

- Chunk level accuracy
- Subject level accuracy
- Mixed accuracy
- Sensitivity (=Recall, True Positive Rate) = Measures how well the model identifies positive cases
  - "Of all actual positives, how many did we correctly identify?"
- Specificity - how well does the model classify negative values =  TN / (TN + FP)
- Precision = True Positives / True positives + False positives
- F1 Score - **multiple classes**, useful for imbalanced datasets
  - F1 Score = 2 x Sensitivity x Precision / (Sensitivity + Precision)
  - Multiclass -> Macro F1 = Calculate F1 for each class separately, then average them (treats all classes equally)
- Confusion matrix - Ideally somehow aggregate all folds, maybe average it out

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
