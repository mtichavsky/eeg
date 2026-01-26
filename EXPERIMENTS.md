# EXPERIMENTS

Run something to figure out impact of adding weights to cross entropy. 

```bash
python main.py train --skip-ica --channel all --batch-size 64 --dataset both --model SmallerAll --condition ec+eo --n-folds 6 --dropout 0.1 --weight-decay 1e-4 --val-every 1 --checkpoint-dir=experiments/both-testing01
```
