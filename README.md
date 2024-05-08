# DL_project

## Setting up
### python env
Create conda environment from `env.yaml`. 

### R env
Since CELESTA is an R package you need to also install R and some of its packages, namely: 
- CELESTA, 
- Rmixmod, 
- spdep, 
- ggplot2, 
- reshape2,  
- zeallot 

Running the script `install_packages.R` should install those packages, given all the dependencies outside R are met.

## EDA

Look into `eda_*` notebooks  

## CELESTA

CELESTA needs two inputs: 
1. Handcrafted marker expression signature file
2. CSV file with expression levels for each cell.

We provide the marker expression signature file, and the CSV files are exported from `.h5ad` file. 

For running CELESTA one needs to run `celesta.py`

``` 
usage: celesta.py [-h] anndata input_dir marker_info_path output_dir

Prepares input suitable for CELESTA based on h5ad file splitting data it into csv files corresponding to pictures
and ROI, as celesta uses X and Y information). Runs celesta and merges the outputs - merged output will be in the
output directory in the file named 'merged_celesta_output.csv'.

positional arguments:
  anndata           path to h5ad file
  input_dir         path to the directory csv files (celesta inputs) will be written to
  marker_info_path  path to csv file with prior marker info (celesta input)
  output_dir        path to the directory celesta output files will be written to

options:
  -h, --help        show this help message and exit

```

## Baselines
### Training
To train the baselines one needs to run `DL4DS_Team_project_Baselines.py` script. 
```
usage: DL4DS_Team_Project_Baselines.py [-h] [--path_to_folder PATH_TO_FOLDER]
                                       [--test_split] [--seed SEED]

Runs baseline training and classification on training data.

options:
  -h, --help            show this help message and exit
  --path_to_folder PATH_TO_FOLDER, -p PATH_TO_FOLDER
                        Path to folder with data
  --test_split          Flag if you want to run validation split
  --seed SEED, -s SEED  Random seed
```

This script: 
 - Trains the baselines
 - Returns classification results and plots confusion matrices for classifiers if `--test_split` flag is used
 - Saves the baselines as pickle files named `baseline_i_ClassOfBaseline.pkl` where i is one of {1, 2, 3} and
ClassOfBaseline is one of {LogisticRegression, GradientBoostingClassifier, MLPClassifier} 
### Prediction
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