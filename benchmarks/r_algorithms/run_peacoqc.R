#!/usr/bin/env Rscript

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Usage: Rscript run_peacoqc.R <input_fcs> <output_fcs>")
}

input_file <- args[1]
output_file <- args[2]

# Load libraries
suppressPackageStartupMessages({
    library(flowCore)
    library(PeacoQC)
})

# Start timing
start_time <- Sys.time()

tryCatch({
    # Read FCS file
    cat("Reading FCS file...\n")
    ff <- read.FCS(input_file, transformation = FALSE)
    initial_cells <- nrow(ff)
    
    # Define channels for quality control
    # Use all channels except Time (usually channel 1)
    all_channels <- 1:ncol(ff)
    # Identify time channel
    time_channel <- which(tolower(colnames(ff)) %in% c("time", "t"))
    if (length(time_channel) > 0) {
        channels <- setdiff(all_channels, time_channel)
    } else {
        channels <- all_channels[-1]  # Assume first channel is Time
    }
    
    # Remove margin events
    cat("Removing margin events...\n")
    ff_margins <- RemoveMargins(ff = ff, channels = channels, output = "frame")
    
    # Check if compensation is available
    spill <- keyword(ff)$SPILL
    if (!is.null(spill) && !is.na(spill)) {
        cat("Compensating data...\n")
        ff_comp <- compensate(ff_margins, spill)
        
        # Transform data
        cat("Transforming data...\n")
        trans <- estimateLogicle(ff_comp, colnames(spill))
        ff_trans <- transform(ff_comp, trans)
    } else {
        cat("No compensation matrix found, proceeding without compensation...\n")
        ff_trans <- ff_margins
    }
    
    # Run PeacoQC
    cat("Running PeacoQC...\n")
    peacoqc_res <- PeacoQC(ff_trans, 
                          channels,
                          determine_good_cells = "all",
                          save_fcs = FALSE,  # We'll save manually
                          output_directory = ".")
    
    # Extract good cells using original (non-transformed) data
    good_cells <- peacoqc_res$GoodCells
    ff_clean <- ff[good_cells, ]
    
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
    cat(sprintf("METRICS|algorithm:peacoqc|initial_cells:%d|final_cells:%d|cells_removed:%d|percent_removed:%.2f|runtime:%.2f\n",
                initial_cells, final_cells, cells_removed, percent_removed, runtime))
    
}, error = function(e) {
    cat(sprintf("ERROR|%s\n", e$message))
    quit(status = 1)
})