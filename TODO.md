# TODO

## Open Questions

- [ ] How long recordings are needed for X% validation accuracy? (plot in evaluation)
- [ ] Is single-channel preprocessing (chunk-level) better than multichannel preprocessing for online/in-ear use case?
- [ ] Spectrogram dimensions don't match paper's (254, 342) — paper may use different windowing (Kaiser window of size 2n-1)

## Preprocessing

- [ ] CANE spectrogram contains frequencies >70 Hz despite bandpass filtering — investigate poor filtering
- [ ] Crop spectrogram frequency axis at 70 Hz to avoid feeding unused bins to the network

## Future Preprocessing Improvements (CANE)

- ICA (Independent Component Analysis) — most effective but expensive
- Wavelet denoising — preserves transients better than simple filtering
- ASR (Artifact Subspace Reconstruction) — state-of-the-art continuous artifact removal
- Bad channel detection and interpolation
- EOG/EMG regression (if reference channels available)

## Experiments

- [ ] Data augmentation for EEG — what types are effective? (performance should stay similar for robust model)
- [ ] Check [Claude suggestions for overfitting](https://claude.ai/share/8b2c46f9-19d3-411f-acac-ae3c87e180ba)