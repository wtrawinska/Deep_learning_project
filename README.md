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
Usage example:
```
python3 celesta.py ~/Downloads/cell_data.h5ad data/celesta_input_final data/final_signature.csv celesta_out
```

To get the metrics for CELESTA output, one needs to run `celesta_output_metrics.py`

```
usage: celesta_output_metrics.py [-h] celesta_output

Calculates metrics for celesta output evaluation.

positional arguments:
  celesta_output  path to merged celesta outputs

options:
  -h, --help      show this help message and exit
```
### Metrics calculated for CELESTA

Prediction results for CELESTA:
1. F1 macro score: 0.3459172731647878
2. Accuracy score: 0.5876405775557348

3. Per type accuracy:
    - B: 0.6886641830761043
    - BnT: 0.0
    - CD4: 0.44647641234711705
    - CD8: 0.5706902696922894
    - DC: 0.1356437915997246
    - HLADR: 0.75625
    - MacCD163: 0.27277628032345014
    - Mural: 0.788136285644252
    - NK: 0.27967479674796747
    - Neutrophil: 0.06343656343656344
    - Treg: 0.6409605622803592
    - Tumor: 0.8855220351638193
    - pDC: 0.36325678496868474
    - plasma: 0.5718071890011596

4. Recall score: 0.3378122372524154

5. ROC AUC per cell type score:
    - B: 0.7614760866005604
    - BnT: 0.647494130923578
    - CD4: 0.47577485424605276
    - CD8: 0.5008067070570508
    - DC: 0.7174014319505817
    - HLADR: 0.4912937753712475
    - MacCD163: 0.49389007305465704
    - Mural: 0.4993243935850006
    - NK: 0.4994547437295529
    - Neutrophil: 0.49172526937709793
    - Treg: 0.2868509539766997
    - Tumor: 0.39923771514231404
    - pDC: 0.6374965280553889
    - plasma: 0.7189946850296984
   
6. Precision: 0.5094310582301507
    
7. Per type precision:
    - B: 0.5443836769036601
    - BnT: nan
    - CD4: 0.4161328845945066
    - CD8: 0.7900493421052631
    - DC: 0.3476470588235294
    - HLADR: 0.12491397109428769
    - MacCD163: 0.5686458138228132
    - Mural: 0.7074440777411074
    - NK: 0.593103448275862
    - Neutrophil: 0.9883268482490273
    - Treg: 0.801513671875
    - Tumor: 0.9044303222877894
    - pDC: 0.05301645338208409
    - plasma: 0.8018583042973287
    
95.2 % of BnT cells were predicted as B cells or T cells

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
