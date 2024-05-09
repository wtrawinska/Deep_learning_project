import argparse
import itertools
#!pip install ann
#!pip install delayedarray
#!pip install pyometiff

import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, \
    recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import anndata
import pickle


def cv(dataset, classifs, num_splits=1, predicts='type', test_set=None, random_seed=1, test_par=False):
    """
    This function trains and validates the classification models on a dataset, with possibility of predicting on a test set, as well as conducting a cross validation.
    :param dataset: Training dataset in the form of pd.DataFrame
    :param classifs: List of classification models to be trained and validated
    :param num_splits: number of splits to use in cross validation procedure, defaults to 1 in simple train - test split (80/20)
    :param predicts: name of columns with feature to be predicted by models, defaults to 'type'
    :param test_set: Test dataset to be used for validation, defaults to None, in the form of pd.DataFrame
    :param random_seed: random seed to be set
    :return:  `mxs` - list of confusion matrices, `rets` - list of classifiers
    """

    np.random.seed(random_seed)  # Set random seed

    print('#' * 160)
    if test_par:
        msk = (np.random.rand(len(dataset)) < .8) if test_set is None else (np.random.rand(len(dataset)) < .0)
        valid = dataset[~msk]
        dataset = dataset[msk]
        if test_set is not None:
            valid = test_set
    mxs = []
    rets = []

    for classif in classifs:
        ret = []
        clfs = []
        print(classif)

        for _ in range(num_splits):
            classifier = classif  # (**arg)

            msk = np.random.rand(len(dataset)) < 1 / num_splits
            train = dataset[msk]
            test = dataset[~msk]
            df = train

            X, y = df.loc[:, df.columns != predicts], df[predicts]

            X_train, X_test, y_train, y_test = (X, test.loc[:, df.columns != predicts], y, test[predicts])

            classifier.fit(X_train, y_train)
            if num_splits > 1:
                y_pred = classifier.predict_proba(X_test)
                try:
                    roc = roc_auc_score(y_test, y_pred, multi_class='ovr')
                except ValueError as e:
                    roc = 0
                    raise e
                ret.append(roc)
            else:
                y_pred = []
                ret = [1]
            clfs.append((classifier, y_test, y_pred))
        y_pred = None
        y_pred_proba = None
        clf = clfs[np.argmax(ret)][0]
        rets.append(clf)
        if test_par or test_set is not None:
            X, y = valid.loc[:, valid.columns != predicts], valid.loc[:, predicts]
            if num_splits > 1:
                print(f"mean ROC: {np.mean(ret)}, std: {np.std(ret)}")
                print(f"maximum ROC: {np.max(ret)} ")
                print("Beneath are the results for the model with the highest ROC")
            y_pred = clf.predict(X)
            y_pred_proba = clf.predict_proba(X)
            print(classification_report(y, y_pred))
            mxs.append(confusion_matrix(y, y_pred, normalize='true'))
            print(f"ROC: {roc_auc_score(y, y_pred_proba, multi_class='ovr')}")
            print(f"Acc: {accuracy_score(y, y_pred)}")
            print(f"Precision: {precision_score(y, y_pred, average='macro')}")
            print(f"Recall: {recall_score(y, y_pred, average='macro')}")
            print(f"Macro F1: {f1_score(y, y_pred, average='macro')}")

        print('#' * 160, '\n')
    return mxs, rets, y_pred_proba, y_pred


def create_data(ann, true_vals=True):
    ann = anndata.read_h5ad(ann)

    if true_vals:
        data_set = np.c_[np.arcsinh(ann.layers['exprs'] / 5.), ann.obs[
            ['area', 'major_axis_length', 'minor_axis_length', 'eccentricity']],  # Training features
        ann.obs['cell_labels'].astype('category')]  # Features to be predicted, changed to int values
        data_set = pd.DataFrame(data_set)
        data_set.columns = [*data_set.columns[:-1],
                            'type']  # Naming the last column as 'type', can be whatever, but must be consistent with `predicts` argument of cv
    else:
        data_set = np.c_[ann.layers['exprs'] , ann.obs[
            ['area', 'major_axis_length', 'minor_axis_length', 'eccentricity']],  # Training features
        ]  # Features to be predicted, changed to int values
        data_set = pd.DataFrame(data_set)
    return ann, data_set


def save_base(bases):
    for i, baseline in enumerate(bases):
        with open(f'baseline_{i + 1}_{type(classifiers[i]).__name__}.pkl', 'wb+') as f:
            pickle.dump(baseline, f)


def load_base(path='baseline_2_GradientBoostingClassifier.pkl'):
    with open(path, 'rb') as f:
        gradient_boost = pickle.load(f)
    return gradient_boost


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs baseline training and classification on training data.")
    parser.add_argument('--path_to_folder', '-p', type=str, default='./', help='Path to folder with data')
    parser.add_argument('--test_split', action='store_true', help="Flag if you want to run validation split")
    parser.add_argument("--seed", '-s', type=int, default=1, help="Random seed")
    args = parser.parse_args()
    # drive.mount('/content/drive') # use if you plan to use colab.
    PATH_TO_FOLDER = args.path_to_folder
    TRAIN_DATA_PATH = 'train'
    ORIGINAL_IMAGE_DATA_SUBDIR = 'images_masks'
    ORIGINAL_MASKS_SUBDIR = 'masks'
    ORIGINAL_IMAGES_SUBDIR = 'img'

    if PATH_TO_FOLDER is None:
        raise ValueError('Please set PATH_TO_FOLDER to a path with unzipped training data.')

    ANNDATA_PATH = 'cell_data.h5ad'
    TRAIN_ANNDATA_PATH = os.path.join(TRAIN_DATA_PATH, ANNDATA_PATH)
    TRAIN_IMAGE_DATA_DIR = os.path.join(TRAIN_DATA_PATH, ORIGINAL_IMAGE_DATA_SUBDIR)
    TRAIN_IMAGE_DATA_IMAGES = os.path.join(TRAIN_IMAGE_DATA_DIR, ORIGINAL_IMAGES_SUBDIR)
    TRAIN_IMAGE_DATA_MASKS = os.path.join(TRAIN_IMAGE_DATA_DIR, ORIGINAL_MASKS_SUBDIR)


    classifiers = [LogisticRegression(max_iter=10000, verbose=False, n_jobs=4), GradientBoostingClassifier(),
                   MLPClassifier(max_iter=10000)]

    train_anndata, data_set = create_data(TRAIN_ANNDATA_PATH)
    matrices, baselines, probs, preds = cv(data_set, classifiers, num_splits=1, random_seed=args.seed, test_par=args.test_split)  # Training and validating

    # Saving baselines:
    save_base(baselines)


    if args.test_split:
        ticks = train_anndata.obs['cell_labels'].astype('category').cat.categories
        fig, axs = plt.subplots(1, len(classifiers))
        for i in range(len(classifiers)):
            im = axs[i].imshow(matrices[i])
            axs[i].set_title(f"{type(classifiers[i]).__name__}")
            axs[i].set_xticks(range(len(ticks)))
            axs[i].set_yticks(range(len(ticks)))
            axs[i].set_xticklabels(ticks, rotation=90)
            axs[i].set_yticklabels(ticks)

        fig.set_figwidth(20)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.show()


    # Example how to use baselines:

    # preds = load_base().predict(data_set.loc[:, data_set.columns != 'type'])
    #
    # print(classification_report(data_set['type'], preds))
    #
