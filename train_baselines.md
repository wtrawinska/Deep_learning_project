# Instruction on how to train the baselines
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