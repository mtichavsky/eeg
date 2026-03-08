

## Test inference CLI:

```bash
python main.py run models/model_inear_2class.pth ../CANE/normals/ec/0015_EC.csv --model Smaller --channel in-ear --class 2
python main.py run models/model_inear_4class.pth ../MDD/MDD\ S1\ EO.edf --model Smaller --channel in-ear --class 4
* python main.py run models/model_8channel_2class.pth ../MDD/MDD\ S1\ EO.edf --model Smaller --channel in-ear --class 2
* python main.py run models/model_8channel_4class.pth ../MDD/MDD\ S1\ EO.edf --model Smaller --channel in-ear --class 4
```


# Test training
CUDA_VISIBLE_DEVICES=0 EEG_DATA_DIR=/home/xticha09 python main.py train   --channel in-ear   --batch-size 64   --dataset all   --model SmallerAttn   --condition ec+eo   --n-folds 10   --dropout 0.1   --weight-decay 1e-4   --val-every 1   --checkpoint-dir=experiments/all3-018-inear-binary-smallerAttn