library(CELESTA)
library(Rmixmod)
library(spdep)
library(ggplot2)
library(reshape2)
library(zeallot)

run_celesta <- function(marker_info_file, imaging_data_file, proj_title=NULL,
                        filter_cells=FALSE,
                        max_iteration=10,cell_change_threshold=0.01,
                        high_expression_threshold_anchor=rep(0.5, length=13),
                        low_expression_threshold_anchor=rep(0.9, length=13),
                        high_expression_threshold_index=rep(0.4, length=13),
                        low_expression_threshold_index=rep(1, length=13)) {
  prior_marker_info <- read.csv(marker_info_file)
  imaging_data <- read.csv(imaging_data_file)
  colnames(prior_marker_info)[1] <-""
  if(is.null(proj_title)) proj_title <- tail(strsplit(strsplit(imaging_data_file, "[.]")[[1]][1], "[/]")[[1]], n=1)
  CelestaObj <- CreateCelestaObject(project_title = proj_title,prior_marker_info,imaging_data)
  if(filter_cells){CelestaObj <- FilterCells(CelestaObj)}
  CelestaObj <- AssignCells(CelestaObj,
                            max_iteration=max_iteration,
                            cell_change_threshold=cell_change_threshold,
                            high_expression_threshold_anchor=high_expression_threshold_anchor,
                            low_expression_threshold_anchor=low_expression_threshold_anchor,
                            high_expression_threshold_index=high_expression_threshold_index,
                            low_expression_threshold_index=low_expression_threshold_index)  
  return (CelestaObj)
}

celesta_output_summary <- function(img_file_path) {
  celesta_out <- paste(tail(strsplit(strsplit(img_file_path, "[.]")[[1]][1], "[/]")[[1]], n=1),"final_cell_type_assignment.csv",sep="_")
  celesta_out <- read.csv(celesta_out)
  head(celesta_out)
  print("Assigned cells stats")
  table(celesta_out$Final.cell.type)
}

#setwd(getSrcDirectory(function(){})[1])
#getwd()

args <- commandArgs(trailingOnly = TRUE)

input_dir <- args[1]
marker_info_path <- args[2]
output_dir <- args[3]

# marker_info_path <- "data/marker_info_data/markers_table_same_round_halfs.csv"

files <- list.files(path=input_dir, pattern="*.csv", full.names=TRUE, recursive=FALSE)

dir.create(output_dir, showWarnings = FALSE)

for (f in files){
  # print(tail(strsplit(strsplit(f, "[.]")[[1]][1], "[/]")[[1]],n=1))
  proj_path <- paste(output_dir, tail(strsplit(strsplit(f, "[.]")[[1]][1], "[/]")[[1]],n=1),sep="/")
  print(f)
  run_celesta(marker_info_path, f, proj_title = proj_path)
}

