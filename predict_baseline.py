import argparse

import numpy as np
import pandas as pd

from DL4DS_Team_Project_Baselines import load_base, create_data
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)


def main(model, data_path, true_vals):
    model = load_base(model)
    train_anndata, data = create_data(data_path, true_vals=true_vals)

    if true_vals:
        data, y_true = data.loc[:, data.columns != 'type'], data.loc[:, 'type']

    y_pred = model.predict(data)
    y_pred_proba = model.predict_proba(data)

    if true_vals:
        print(f"Prediction results for data in {data_path}:")
        print(f"F1 macro score: {f1_score(y_true, y_pred, average='macro')}")
        print(f"Accuracy score: {accuracy_score(y_true, y_pred)}")
        matrix = confusion_matrix(y_true, y_pred)
        print("#############")
        print(f"Per type accuracy:")
        for i in [f'{model.classes_[i]}: {x}' for i, x in enumerate(matrix.diagonal() / matrix.sum(axis=1))]:
            print(i)
        print("#############")
        print(f"Precision score: {precision_score(y_true, y_pred, average='macro')}")
        print(f"Recall score: {recall_score(y_true, y_pred, average='macro')}")
        print(f"ROC AUC per cell type score:")
        for i in [f'{model.classes_[i]}: {x}' for i, x in enumerate(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=None))]:
            print(i)
        print(f"Average Precision  per cell type score:")
        for i in [f'{model.classes_[i]}: {x}' for i, x in enumerate(average_precision_score(y_true, y_pred_proba, average=None))]:
            print(i)
    return y_pred, y_pred_proba, model.classes_


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Runs prediction of a given model (implementing scipy estimator"
                                                     " interface and saved in pickle) and returns prediction scores"
                                                     " if true labels are known. Also, saves prediction results in "
                                                     " a csv file.")
    arg_parser.add_argument('--path_to_model', type=str, default='./baseline_2_GradientBoostingClassifier.pkl',
                            help='Path to saved model')
    arg_parser.add_argument('--path_to_data', type=str, default='./train/cell_data.h5ad', help='Path to data file')
    arg_parser.add_argument('--no_labels', action='store_false', help="Flag if there is no labels in the dataset.")
    arg_parser.add_argument("--save_path", type=str, default='', help='Path to save prediction results as '
                                                                      'csv, if not given the results are not saved')

    args = arg_parser.parse_args()

    pred, probs, var = (main(args.path_to_model, args.path_to_data, args.no_labels))

    res = pd.DataFrame(data=np.c_[pred, probs], columns=['final_prediction', *[f"{x}_probability" for x in var]])
    print(res)
    if args.save_path != '':
        res.to_csv(args.save_path)
