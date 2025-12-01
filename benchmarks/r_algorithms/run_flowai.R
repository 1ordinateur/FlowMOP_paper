#!/usr/bin/env Rscript

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Usage: Rscript run_flowai.R <input_fcs> <output_fcs>")
}

input_file <- args[1]
output_file <- args[2]

# Load libraries
suppressPackageStartupMessages({
    library(flowCore)
    library(flowAI)
})

# Start timing
start_time <- Sys.time()

tryCatch({
    # Read FCS file
    cat("Reading FCS file...\n")
    ff <- read.FCS(input_file, transformation = FALSE)
    initial_cells <- nrow(ff)
    
    # Run FlowAI quality control
    cat("Running FlowAI...\n")
    ff_clean <- flow_auto_qc(ff, 
                            html_report = FALSE, 
                            mini_report = FALSE, 
                            fcs_QC = FALSE, 
                            folder_results = FALSE)
    
    final_cells <- nrow(ff_clean)
    cells_removed <- initial_cells - final_cells
    percent_removed <- (cells_removed / initial_cells) * 100
    
    # Write cleaned FCS
    cat("Writing cleaned FCS...\n")
    write.FCS(ff_clean, output_file)
    
    # End timing
    end_time <- Sys.time()
    runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    # Output metrics
    cat(sprintf("METRICS|algorithm:flowai|initial_cells:%d|final_cells:%d|cells_removed:%d|percent_removed:%.2f|runtime:%.2f\n",
                initial_cells, final_cells, cells_removed, percent_removed, runtime))
    
}, error = function(e) {
    cat(sprintf("ERROR|%s\n", e$message))
    quit(status = 1)
})