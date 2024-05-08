import argparse
import numpy as np
import os
import pandas as pd

def merge_celesta_output(input_dir, output_dir):

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

    return pd.DataFrame(data_array, columns = columns).sort_values('cell_id')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str,
                        help="path to the directory with celesta input files with expression data")
    parser.add_argument("celesta_output_dir", type=str,
                        help="path to the directory with celesta output csv files")
    parser.add_argument("output_dir", type=str,
                        help="path to the directory to write merged celesta output to")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    merge_celesta_output(args.input_dir, args.celesta_output_dir).to_csv(os.path.join(args.output_dir, "merged_celesta_output.csv"))

if __name__ == "__main__":
    main()

