poetry run ruff format .
The virtual environment found in /home/milan/Documents/notes/tools-notes/diplomka/code/venv seems to be broken.
Recreating virtualenv thesis-MdkZeRIB-py3.13 in /home/milan/.cache/pypoetry/virtualenvs/thesis-MdkZeRIB-py3.13
5 files reformatted, 1 file left unchanged


- How did they work with multi channel? Did they use both eyes open/eyes closed?
- Make it possible to select channels in the dataset
- torch tensor devices
- choose the right channel when working with one only

Via Claude code:

Perfect! So they're also using 10-second segments. Let me calculate the parameters:

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

# Option 1: 1-second window, 75% overlap (more time resolution)
f, t, Zxx = stft(channel, fs=SFREQ, nperseg=250, noverlap=187, window='hamming')

# Option 2: 0.5-second window, 75% overlap (even more time bins)
f, t, Zxx = stft(channel, fs=SFREQ, nperseg=125, noverlap=94, window='hamming')

Would you like me to update your code with these parameters? Which window size do you prefer (0.5 or 1 second)? Or can you share the paper
name so I can check their exact method?


=> made second conv 5x5
