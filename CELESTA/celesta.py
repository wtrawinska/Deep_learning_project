import anndata
import argparse
import numpy as np
import subprocess
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def extract_celesta_input(df, image, roi, dir):
    """
    Extracts columns from a dataframe (concatenated expression and obs from adata)
    and writes csv suitable for celesta input (data for only one image and tissue)
    """
    df_subset = df[(df['image']==image) & (df['ROI']==roi)]
    img_name = image.split(".")[0]
    csv_path = os.path.join(dir,f"{img_name}_roi_{roi}.csv")
    df_subset.to_csv(csv_path, index=False)

    return csv_path


def celesta_input(anndata_path, input_dir):
    """
    Prepares celesta input from anndata (h5ad file)
    """
    adata = anndata.read_h5ad(anndata_path)
    marker_names = list(adata.var['marker'])
    df = pd.DataFrame(adata.layers['exprs'], columns = marker_names)
    df = pd.concat([df,adata.obs[['Pos_X','Pos_Y','cell_labels','ROI','image']].reset_index(drop=True)], axis=1)
    # cell id - row index is added, to properly merge celesta outputs (conserving the cell order)
    df['cell_id'] = df.index
    df.rename(columns={"Pos_Y":"Y", "Pos_X":"X"}, inplace=True)
    celesta_cols = ['X','Y','cell_id','cell_labels','ROI','image'] + marker_names
    df = df[celesta_cols]

    csv_paths = []
    for img in df['image'].unique():
        for roi in df[df['image']==img]['ROI'].unique():
            # print(img, roi)
            csv_paths.append(extract_celesta_input(df, img, roi, input_dir))
    
    return csv_paths

def run_celesta(input_dir, marker_info_path, output_dir):
    """
    Runs celesta on the prepared input
    """

    # Command to run the R script with arguments
    command = ["Rscript", "run_celesta.R", input_dir, marker_info_path, output_dir]

    # Execute the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to finish and get return code
    stdout, stderr = process.communicate()
    return_code = process.returncode

    # Check if the process completed successfully
    if return_code == 0:
        print("R script executed successfully.")
    else:
        print("Error executing R script:")
        print(stderr.decode("utf-8"))

def merge_celesta_output(input_dir, output_dir):
    """
    Merges Celesta outputs so that they are in the same order as in the full anndata file
    Selects only "cell_labels" and "Final cell type" to reduce the size of a file
    """

    celesta_suffix = "_final_cell_type_assignment.csv"
    csv_in_files = os.listdir(input_dir)
    df_in = pd.read_csv(os.path.join(input_dir, csv_in_files[0]))
    df_celesta = pd.read_csv(os.path.join(output_dir, csv_in_files[0].split(".")[0] + celesta_suffix))
    df_merged = pd.concat([df_in[['X','Y','cell_id','cell_labels','ROI','image']].reset_index(drop=True), df_celesta], axis=1)  

    columns = df_merged.columns

    data_array = df_merged.to_numpy()

    for csv_in in csv_in_files[1:]:
        df_in = pd.read_csv(os.path.join(input_dir, csv_in))
        df_celesta = pd.read_csv(os.path.join(output_dir, csv_in.split(".")[0] + celesta_suffix))
        df_merged = pd.concat([df_in[['X','Y','cell_id','cell_labels','ROI','image']].reset_index(drop=True), df_celesta], axis=1)
        data_array = np.append(data_array, df_merged.to_numpy(), axis=0)

    return pd.DataFrame(data_array, columns = columns).sort_values('cell_id')[['cell_id','cell_labels','Final cell type']]

def main():
    parser = argparse.ArgumentParser(description="""Prepares input suitable for CELESTA based on h5ad file
                                     (splitting data it into csv files corresponding to pictures
                                     and ROI, as celesta uses X and Y information). Runs celesta
                                     and merges the outputs - merged output will be in the output directory
                                     in the file named 'merged_celesta_output.csv'. Cells in the merged output
                                     are in the same order as in anndata file.
                                    """)

    parser.add_argument("anndata", type=str,
                        help="path to h5ad file")
    parser.add_argument("input_dir", type=str,
                        help="path to the directory csv files (celesta inputs) will be written to")
    parser.add_argument("marker_info_path", type=str,
                        help="path to csv file with prior marker info (celesta input)")
    parser.add_argument("output_dir", type=str,
                        help="path to the directory celesta output files will be written to")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    anndata = args.anndata
    marker_info_path = args.marker_info_path

    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    
    print("Creating CELESTA input")
    celesta_input(anndata, input_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    print("Running Celesta")
    run_celesta(input_dir, marker_info_path, output_dir)

    print("Merging CELESTA outputs")
    df_merged_output = merge_celesta_output(input_dir, output_dir)
    df_merged_output.to_csv(os.path.join(args.output_dir, "merged_celesta_output.csv"), index=False)

if __name__ == "__main__":
    main()

