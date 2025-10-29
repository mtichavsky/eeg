Ve vlaku:
- [ ] precist ten paper
- [ ] precist Markov - jak resi tyto problemy


Problems bellow - write up about each problem that you can put into your thesis:

- [ ] Window size & combining multiple channels - **TODO** create a short write up why you do it how you do it
- [ ] New architecture to make data loading more modular:
- [ ] I need one subcommand for training and one for running the best model. Also, rewriting the checkpoints? Notify the user
if the directory exists.

Perfect! So they're also using 10-second segments. Let me calculate the parameters:

When training, I wanna focus on labeling each 10sec chunk properly. When evaluating, I wanna focus on labeling
the subject properly too. Let's start by aggregating 1s and 0s.

So far:
- Eyes closed only
- One channel - full remapping of channel, maybe make it possible to pick which ones to use
- Locate the in ear channels

`train --skip-ica` option

Double check that MDD frames are not shorter then Healthy ones
I'd love to do lazy loading. Make sure it didn't see the data.

## Making debugging faster

- use a small subset of data (less subjects)
- skip ICA
- Disk caching

## Window size problem

Window Size Conversion
At SFREQ = 250 Hz:
- 0.2 seconds → nperseg = 50 samples
- 0.5 seconds → nperseg = 125 samples
- 1.0 seconds → nperseg = 250 samples

Your current nperseg=256 is actually ~1 second, which matches the upper range!

The Issue

With 10-second segments, you can't get (254, 342) dimensions using standard STFT. Even with maximum overlap:
- nperseg=250 → 126 frequency bins (not 254)
- 10 seconds → ~21-80 time bins depending on overlap (not 342)

I suspect the paper either:
1. Stacks multiple channels to create a multi-channel "image"
2. Uses a different spectrogram method
3. Has a typo in their dimensions

My Recommendation

Use the paper's window size range with high overlap:

### Option 1: 1-second window, 75% overlap (more time resolution)
f, t, Zxx = stft(channel, fs=SFREQ, nperseg=250, noverlap=187, window='hamming')

### Option 2: 0.5-second window, 75% overlap (even more time bins)
f, t, Zxx = stft(channel, fs=SFREQ, nperseg=125, noverlap=94, window='hamming')

Would you like me to update your code with these parameters? Which window size do you prefer (0.5 or 1 second)? Or can you share the paper
name so I can check their exact method?

=> made second conv 5x5

Short-Time Fourier Transform (STFT) -> spectogram


## New architecture to make data loading more modular:

Current Architecture Issues:
- SpectrogramDataset is both a dataset and a preprocessing pipeline
- It's tightly coupled to the DataLoader pattern
- Hard to reuse for different models with different preprocessing needs

Better Architecture (for future):

I'd recommend a transform function approach (like PyTorch's torchvision):

# Preprocessing pipelines as composable transforms
def to_spectrogram(eeg_chunk, nperseg=256, noverlap=192):
"""Transform raw EEG to spectrogram"""
# ... STFT logic
return spec_tensor

def to_wavelet(eeg_chunk, wavelet='db4', levels=5):
"""Transform raw EEG to wavelet decomposition"""
# ... wavelet logic
return wavelet_tensor

# Usage
dataset = MDDDataset(
condition="EC",
transform=to_spectrogram  # or to_wavelet, or None for raw
)

Why this is better:
- Each model defines its own transform function
- MDDDataset stays generic (just loads raw EEG)
- Easy to A/B test different preprocessing approaches
- Composable: transform=lambda x: to_spectrogram(normalize(x))

For your MVP right now:

Let's stick with option #3 to fix the immediate performance issue. You can refactor to transforms later when you add more models. The quick
fix won't prevent you from doing this refactor later.

Implementation plan:
1. Create one SpectrogramDataset from ALL data (before CV loop)
2. Use torch.utils.data.Subset with fold indices to create train/val splits
3. Spectrograms computed once, reused across all folds

Should I proceed with implementing option #3 now?

## TODO is this implemented this way? WTF is Kaiser window

In this work, an automated depression detection system Dep-
Cap is proposed wherein pre-processed 1D EEG is first
converted into 2D spectrogram images using STFT. Here,
a medium size Kaiser window of size 2n − 1 is used for
windowing, where n is the number of bits.


## Evaluation 

How They Evaluate (Section V.B, Page 8)

  Key finding: They use 10-fold cross-validation but evaluate on CHUNK-level predictions, not subject-level aggregation.

  Here's what they do:

  Their Preprocessing (Page 4):

  - 5-minute EEG recordings → segmented into 10-second chunks
  - Each chunk → converted to spectrogram (254×342) via STFT
  - Each spectrogram → fed through CNN-LSTM independently

  Their Evaluation (Page 8, Section V.B):

  "All proposed and pre-trained models are evaluated based on
  performance metrics, including accuracy, specificity, precision,
  and sensitivity employing a 10-fold cross-validation approach"

  Critical issue: They calculate metrics from the confusion matrix of all 141 chunk predictions, not 6 subject predictions!

  From their results:
  - "Fold 1 | Epoch 002/50 | VAL | Confusion: TP=52, TN=87, FP=0, FN=2"
  - That's 141 predictions (52+87+0+2), not 6 subjects!

  The LSTM Architecture Issue You're Raising

  You're absolutely right to question this! Looking at Figure 6 (page 6):

  The LSTM processes spatial features from the CNN for a SINGLE spectrogram, not a temporal sequence of multiple chunks. Each 10-second chunk
  is processed independently:

  10s EEG chunk → Spectrogram → CNN layers → LSTM → Dense → Prediction

  The LSTM is used for "sequence learning" of the frequency bands within one spectrogram, not across multiple time chunks from the same
  subject!

  What They SHOULD Do (Subject-Level Evaluation)

  For proper subject-level evaluation with LSTMs, you have two options:

  Option 1: Sequential LSTM (Using final state)

```
  # Feed all chunks from one subject sequentially
  for subject_chunks in validation_subject_data:
      hidden_state = None
      for chunk in subject_chunks:
          output, hidden_state = lstm(chunk, hidden_state)
      # Use FINAL output for subject prediction
      subject_prediction = classifier(output)  # This is what you're suggesting!
```
  Option 2: Chunk-Level + Aggregation (Simpler)
```
  # Predict each chunk independently, then aggregate
  chunk_predictions = []
  for chunk in subject_chunks:
      pred = model(chunk)  # Independent predictions
      chunk_predictions.append(pred)
```
```
  # Aggregate to subject level (majority vote or probability averaging)
  subject_prediction = majority_vote(chunk_predictions)
```
  The paper does neither! They evaluate on individual chunks, which is why you see:
  - Continuous accuracy values (not 1/6 = 16.67% increments)
  - 98.58% accuracy from 141 predictions, not 6

  Recommendation for Your Implementation

  Given you might want real-time capability, I suggest Option 2 (chunk-level with subject aggregation):

  1. During training: Train on individual chunks (as you're doing now)
  2. During validation: Group predictions by subject and aggregate:
```
  # Evaluate at subject level
  subject_predictions = {}
  for chunk, label, subject_id in val_loader:
      pred = model(chunk)
      if subject_id not in subject_predictions:
          subject_predictions[subject_id] = []
      subject_predictions[subject_id].append(pred)
```
```
  # Aggregate per subject (majority vote or mean probability)
  final_predictions = {}
  for subject_id, preds in subject_predictions.items():
      # Option A: Majority vote
      final_pred = mode(preds)
      # Option B: Average probabilities
      final_pred = (mean(softmax(preds)) > 0.5).astype(int)
      final_predictions[subject_id] = final_pred
```
  # Now calculate metrics on 6 subject predictions

  This gives you the 1/6 increment accuracy you expect AND supports real-time use (you can make predictions on individual chunks, but validate
   on aggregated subject-level performance).

  Want me to implement proper subject-level evaluation for your code?
  


# Key decisions - how to evaluate, should LSTM work on the whole 5 minute segment or should the memory refresh every 10sec

How to measure the accuracy will be definitely a key decision.

Maybe try both, I think with the speed it's going it's more interesting.
You definitely need to care about more then 1 result, otherwise you'd have hard time making any training progress,
so definitely results as they go
but I'm interested how it interpolates going from 10 sec chunks to 5 minute chunks
Maybe it's okay to aggregate 1s and 0s to get the result, maybe it's better to run in on the whole 5 minute times
Or

Chunk-Level Evaluation (Current)

  What it measures: "Given a random 10-second EEG segment, can I classify it as MDD/Healthy?"

  Problems:
  1. Artificially inflated accuracy - Multiple chunks from same subject are highly correlated (same brain, same recording session)
  2. Data leakage - If the model learns subject-specific patterns (like "Subject MDD S31 has distinctive alpha asymmetry"), it correctly
  classifies all ~24 chunks from that subject
  3. Doesn't answer the clinical question - You don't treat individual 10-second windows, you treat patients

  When it makes sense:
  - If you're studying "what EEG patterns in 10-second windows are indicative of MDD"
  - If you have truly independent chunks (different sessions, different days)

  Subject-Level Evaluation (Recommended)

  What it measures: "Given a new patient's EEG recording, can I diagnose them with MDD?"

  Advantages:
  1. Honest performance estimate - Reflects ability to generalize to new patients
  2. Clinically relevant - Matches real-world use case
  3. Conservative - Forces model to learn generalizable MDD biomarkers, not subject-specific quirks
  4. Literature standard - Most reputable EEG-based disorder papers use subject-level metrics

  How to implement:
  # During validation, aggregate chunk predictions per subject
  subject_votes = {}
  for chunk, label, subject_id in val_loader:
      pred = model(chunk)
      if subject_id not in subject_votes:
          subject_votes[subject_id] = {'preds': [], 'true_label': label}
      subject_votes[subject_id]['preds'].append(pred)

  # Majority vote per subject
  subject_predictions = []
  subject_labels = []
  for subj_id, data in subject_votes.items():
      # Majority vote
      final_pred = mode(data['preds'])  # or np.mean() > 0.5 for probabilities
      subject_predictions.append(final_pred)
      subject_labels.append(data['true_label'])

  # Calculate metrics on 6 subjects (not 141 chunks)
  metrics = classification_metrics(subject_labels, subject_predictions)
  # Now accuracy will be in 1/6 increments: 0%, 16.67%, 33.33%, ..., 100%

  My Recommendation

  Use subject-level evaluation for validation because:
  - Your research question is likely "Can we detect MDD in patients?"
  - Your current 98.58% might drop to ~80-85% (more realistic)
  - Reviewers will question chunk-level metrics as inflated
  - Subject-level splitting only makes sense if you evaluate at subject-level

  BUT you can still:
  - Train on individual chunks (efficient, more training samples)
  - Make real-time chunk-level predictions during inference
  - Just aggregate during validation to get honest performance

  The Real-Time Use Case

  Even for real-time, you'd likely want:
  # Real-time: Stream of predictions over time
  chunk_preds = [0, 1, 1, 1, 0, 1, 1, 1, ...]  # Over 5 minutes
  # Clinical decision: Aggregate over session
  final_diagnosis = majority_vote(chunk_preds)

  Want me to implement subject-level validation in your main.py? It won't break your real-time capability, just gives you honest metrics.


