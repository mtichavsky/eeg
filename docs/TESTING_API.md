# API testing & other fun stuff

## Test inference CLI:

Run the server:

```
poetry run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Commands to test:

```bash
python main.py run models/model_inear_binary.pth ../IDUN_IN_EAR/comorbid/0018/eeg_0018EC.csv --model CNNAttn --channel in-ear --class 2
python main.py run models/model_inear_4class.pth ../MDD/MDD\ S1\ EO.edf --model CNNAttn --channel in-ear --class 4
python main.py run models/model_8channel_binary.pth ../SAD/anxious/ec/C1.edf --model AllTransformerV4 --channel all --class 2
python main.py run models/model_8channel_4class.pth ../CANE/normals/ec/0015_EC.csv --model AllTransformerV4 --channel all --class 4
```

To test Raw-EEG model:

```bash
python main.py run ../experiments/all3-1.0-all-binary-lggnet-s/fold_2_best.pth ../MDD/H\ S20\ EC.edf  --model LGGNetS
```

## Test inference API:

```bash
podman run \
    --name eeg-api \
    -v ./models:/app/models:ro,Z \
    -e DEVICE=cpu \
    --network host \
    docker.io/mtichavsky/eeg-classifier-api:latest
```

API:

```bash
# in-ear, 2-class
curl -X POST http://localhost:8000/predict \
  -F "eeg_recording=@../IDUN_IN_EAR/comorbid/0018/eeg_0018EC.csv" \
  -F "electrode_setup=in-ear" \
  -F "classification_task=binary" \
  -F "request_id=$(uuidgen)" \
  -F "user_id=$(uuidgen)"

# in-ear, 4-class
curl -X POST http://localhost:8000/predict \
  -F "eeg_recording=@../MDD/MDD S1 EO.edf" \
  -F "electrode_setup=in-ear" \
  -F "classification_task=4class" \
  -F "request_id=$(uuidgen)" \
  -F "user_id=$(uuidgen)"

# 8-channel, 2-class
curl -X POST http://localhost:8000/predict \
  -F "eeg_recording=@../SAD/anxious/ec/C1.edf" \
  -F "electrode_setup=8channel" \
  -F "classification_task=binary" \
  -F "request_id=$(uuidgen)" \
  -F "user_id=$(uuidgen)"

# 8-channel, 4-class
curl -X POST http://localhost:8000/predict \
  -F "eeg_recording=@../CANE/normals/ec/0015_EC.csv" \
  -F "electrode_setup=8channel" \
  -F "classification_task=4class" \
  -F "request_id=$(uuidgen)" \
  -F "user_id=$(uuidgen)"
```


# Test training
CUDA_VISIBLE_DEVICES=0 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel in-ear   --batch-size 64   --dataset all   --model CNNLSTMAttn   --condition ec+eo   --n-folds 10   --dropout 0.1   --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-018-inear-binary-smallerAttn