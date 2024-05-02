library(CELESTA)
library(Rmixmod)
library(spdep)
library(ggplot2)
library(reshape2)
library(zeallot)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
print(getwd())
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")

run_celesta <- function(marker_info_file, imaging_data_file) {
  prior_marker_info <- read.csv(marker_info_file)
  imaging_data <- read.csv(imaging_data_file)
  colnames(prior_marker_info)[1] <-""
  CelestaObj <- CreateCelestaObject(project_title = strsplit(marker_info_file, "[.]")[[1]][1],prior_marker_info,imaging_data)
  CelestaObj <- FilterCells(CelestaObj,high_marker_threshold=0.9, low_marker_threshold=0.4)
  
  CelestaObj <- AssignCells(CelestaObj,max_iteration=10,cell_change_threshold=0.01,
                            high_expression_threshold_anchor=high_marker_threshold_anchor,
                            low_expression_threshold_anchor=low_marker_threshold_anchor,
                            high_expression_threshold_index=high_marker_threshold_iteration,
                            low_expression_threshold_index=low_marker_threshold_iteration)
}

img_path <- "data/exprs_data/IMMUcan_Batch20220908_S-220729-00002_002_roi_2.csv"
marker_info_path <- "data/markers_table.csv"

run_celesta(marker_info_path, img_path)

