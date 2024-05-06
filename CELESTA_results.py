import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import os
import anndata
import seaborn as sns


PATH_TO_FOLDER = './'
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

train_anndata = anndata.read_h5ad(TRAIN_ANNDATA_PATH)
test = train_anndata.obs[train_anndata.obs["image"] == "IMMUcan_Batch20220908_S-220729-00002_002.tiff"]

output = pd.read_csv('CELESTA/thresholds_1_final_cell_type_assignment.csv')[["X", "Y", "Final cell type"]]

test_chosen = test[["Pos_X", "Pos_Y", "cell_labels"]]
test_chosen["output"] = output["Final cell type"].tolist()

print(f"{f1_score(test_chosen['cell_labels'], test_chosen['output'], average='macro'):.4f}", "f1 macro")
print(f"{f1_score(test_chosen['cell_labels'], test_chosen['output'], average='weighted'):.4f}", "f1 weighted")
print(f"acc: {accuracy_score(test_chosen['cell_labels'], test_chosen['output']):.4f} ")
confusion_matrix = confusion_matrix(test_chosen['cell_labels'], test_chosen['output'], normalize='true')
print(confusion_matrix)
# plt.matshow(confusion_matrix)

# plt.show()
ticks = train_anndata.obs['cell_labels'].astype('category').cat.categories
ax = plt.imshow(confusion_matrix) #, annot=True, fmt='.1f', xticklabels=ticks, yticklabels=ticks)

#
ax.set_xticks(range(len(ticks)))
ax.set_yticks(range(len(ticks)))
ax.set_xticklabels(ticks, rotation=90)
ax.set_yticklabels(ticks, rotation=0)

plt.show()
print("Counts for test:")
print(test_chosen[['output', 'cell_labels']].value_counts())
