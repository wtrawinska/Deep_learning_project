import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

def evaluate_celesta_output(csv_file):
    """
    calculates metrics for celesta classifcation
    """
    df = pd.read_csv(csv_file)
    y_pred = df['Final cell type']
    y_true = df['cell_labels']
    print(f"Prediction results for CELESTA:")
    print(f"F1 macro score: {f1_score(y_true, y_pred, average='macro')}")
    print(f"Accuracy score: {accuracy_score(y_true, y_pred)}")
    matrix = confusion_matrix(y_true, y_pred)
    cell_labels = y_true.unique()
    cell_labels.sort()
    matrix = confusion_matrix(y_true, y_pred, labels=cell_labels)
    matrix_df = pd.DataFrame(matrix, columns = cell_labels)
    matrix_df["true"] = cell_labels
    matrix_df.set_index("true", inplace=True)
    print("#############")
    print(f"Per type accuracy:")
    for i in [f'{cell_labels[i]}: {x}' for i, x in enumerate(matrix.diagonal() / matrix.sum(axis=1))]:
        print(i)
    print("#############")
    print(f"Precision score: {precision_score(y_true, y_pred, average='macro')}")
    print("#############")
    print(f"Per type precision:")
    for i in [f'{cell_labels[i]}: {x}' for i, x in enumerate(matrix.diagonal() / matrix.sum(axis=0))]:
        print(i)
    print("#############")
    print(f"Recall score: {recall_score(y_true, y_pred, average='macro')}")
    y_score = OneHotEncoder().fit_transform(y_pred.to_numpy().reshape(-1, 1)).toarray()
    print(f"ROC AUC per cell type score:")
    for i in [f'{cell_labels[i]}: {x}' for i, x in enumerate(roc_auc_score(y_true, y_score, multi_class='ovr', average=None, labels=list(cell_labels)))]:
        print(i)
    print("#############")
    print(f'{round(matrix_df.loc["BnT", ["B", "CD4", "CD8", "Treg"]].sum() / matrix_df.loc["BnT", :].sum() * 100, 2)} % of BnT cells were predicted as B cells or T cells')

def main():
    parser = argparse.ArgumentParser(description="""
                                    Calculates metrics for celesta output evaluation.
                                    """)
    parser.add_argument("celesta_output", type=str,
                        help="path to merged celesta outputs")
    args = parser.parse_args()

    evaluate_celesta_output(args.celesta_output)    

if __name__ == "__main__":
    main()