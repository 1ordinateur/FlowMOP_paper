#!/usr/bin/env Rscript

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# If no arguments, process all FCS files in current directory
if (length(args) == 0) {
    # Running from RStudio - process all FCS files in working directory
    input_files <- list.files(pattern = "\\.fcs$", ignore.case = TRUE, full.names = TRUE)
    if (length(input_files) == 0) {
        stop("No FCS files found in current directory")
    }
    output_dir <- file.path(getwd(), "_flowcut")
} else {
    # Command line mode - single file
    if (length(args) < 1) {
        stop("Usage: Rscript run_flowcut.R <input_fcs> [output_dir]")
    }
    input_files <- args[1]
    
    # Use provided output directory or create default
    if (length(args) >= 2) {
        output_dir <- args[2]
    } else {
        # Default: current working directory + '_flowcut'
        output_dir <- file.path(getwd(), "_flowcut")
    }
}

# Create output filename in the specified directory
input_basename <- basename(input_file)
output_filename <- sub("\\.[^.]*$", "_flowcut.fcs", input_basename)
output_file <- file.path(output_dir, output_filename)

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

# Load libraries
suppressPackageStartupMessages({
    library(flowCore)
    library(flowCut)
})

# Start timing
start_time <- Sys.time()

tryCatch({
    # Read FCS file
    cat("Reading FCS file...\n")
    ff <- read.FCS(input_file, transformation = FALSE)
    initial_cells <- nrow(ff)
    
    # Run flowCut
    cat("Running flowCut...\n")
    res_flowCut <- flowCut(ff, 
                          Plot = "None",  # Disable plotting for batch processing
                          PrintToConsole = FALSE,
                          AllowFlaggedRerun = TRUE,
                          UseOnlyWorstChannels = TRUE)
    
    # Extract cleaned data
    if (!is.null(res_flowCut$ind)) {
        ff_clean <- ff[res_flowCut$ind, ]
    } else {
        # If no indices returned, assume all cells are good
        ff_clean <- ff
    }
    
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
    cat(sprintf("METRICS|algorithm:flowcut|initial_cells:%d|final_cells:%d|cells_removed:%d|percent_removed:%.2f|runtime:%.2f\n",
                initial_cells, final_cells, cells_removed, percent_removed, runtime))
    
}, error = function(e) {
    cat(sprintf("ERROR|%s\n", e$message))
    quit(status = 1)
})