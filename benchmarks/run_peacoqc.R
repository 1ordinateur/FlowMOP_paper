# Basic PeacoQC Batch Processing - Just paste and run!

# Load libraries
library(flowCore)
library(PeacoQC)

# Get all .fcs files in current directory
fcs_files <- list.files(".", pattern = "\\.fcs$", full.names = TRUE, ignore.case = TRUE)

if (length(fcs_files) == 0) {
  stop("No .fcs files found in current directory")
}

cat("Found", length(fcs_files), ".fcs files. Processing...\n")

# Process each file
for (i in seq_along(fcs_files)) {
  file_path <- fcs_files[i]
  file_name <- basename(file_path)
  file_base <- tools::file_path_sans_ext(file_name)
  
  cat("Processing", i, "of", length(fcs_files), ":", file_name, "...")
  
  # Read file
  ff <- read.FCS(file_path)
  
  # Get all channels except scatter and time channels for QC
  all_channels <- colnames(ff)
  # Exclude typical scatter channels (FSC, SSC) and Time
  qc_channels <- which(!grepl("FSC|SSC|Time", all_channels, ignore.case = TRUE))
  
  # If no channels found, use all channels
  if (length(qc_channels) == 0) {
    qc_channels <- 1:ncol(ff)
  }
  
  # Run PeacoQC (minimal parameters, no plots, no reports)
  result <- PeacoQC(ff = ff, 
                    channels = qc_channels,
                    plot = FALSE, 
                    save_fcs = TRUE, 
                    report = FALSE,
                    output_directory = ".",
                    name_directory = ".",
                    suffix_fcs = "_peacoqc")
  
  cat(" Done\n")
}

cat("All files processed! Cleaned files saved with '_peacoqc' suffix.\n")