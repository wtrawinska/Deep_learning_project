import anndata
import argparse
import subprocess
import os
import pandas as pd

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

    Parameters
    ==========
    anndata_path: str
        path to h5ad file

    input_dir: str
        output directory for csv files - celesta's inputs

    Returns
    =======
    csv_paths: List[str]
        paths to created csv files
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("anndata", type=str,
                        help="path to h5ad file")
    # parser.add_argument("marker_info", type=str,
    #                     help="path to csv file with prior marker info (celesta input)")
    parser.add_argument("input_dir", type=str,
                        help="path to the directory csv files (celesta inputs) will be written to")
    # parser.add_argument("output_dir", type=str,
    #                     help="path to the directory celesta output files will be written to")
    args = parser.parse_args()

    # CELESTA_INPUT_DIR = "celesta_input"
    # CELESTA_OUTPUT_DIR = "celesta_output"

    if not os.path.exists(args.input_dir):
        os.mkdir(args.input_dir)
    
    celesta_input(args.anndata, args.input_dir)

    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.input_dir)
    
    # for csv_file in os.listdir(args.input_dir):
    #     csv_path = os.path.join(args.input_dir,csv_path)
    #     a = "a"
        # subprocess.call(["Rscript", "test.R", "a"])

if __name__ == "__main__":
    main()

