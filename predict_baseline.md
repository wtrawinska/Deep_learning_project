# Instruction on how to run prediction with baselines
To run prediction on the test set one needs to run script `predict_baseline.py`
```
usage: predict_baseline.py [-h] [--path_to_model PATH_TO_MODEL]
                           [--path_to_data PATH_TO_DATA] [--no_labels]
                           [--save_path SAVE_PATH]

Runs prediction of a given model (implementing scipy estimator interface and
saved in pickle) and returns prediction scores if true labels are known. Also,
saves prediction results in a csv file.

options:
  -h, --help            show this help message and exit
  --path_to_model PATH_TO_MODEL
                        Path to saved model
  --path_to_data PATH_TO_DATA
                        Path to data file
  --no_labels           Flag if there is no labels in the dataset.
  --save_path SAVE_PATH
                        Path to save prediction results as csv, if not given
                        the results are not saved

```

This script: 
 - Runs the prediction of one baseline
 - Prints classification results for baseline if true labels are known
 - Saves the baseline's predictions as csv file specified by `--save_path` parameter 