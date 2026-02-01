

## Test inference CLI:

```bash
python main.py run models/model_inear_2class.pth ../CANE/normals/ec/0015_EC.csv --model Smaller --channel in-ear --class 2
python main.py run models/model_inear_4class.pth ../MDD/MDD\ S1\ EO.edf --model Smaller --channel in-ear --class 4
* python main.py run models/model_8channel_2class.pth ../MDD/MDD\ S1\ EO.edf --model Smaller --channel in-ear --class 2
* python main.py run models/model_8channel_4class.pth ../MDD/MDD\ S1\ EO.edf --model Smaller --channel in-ear --class 4
```