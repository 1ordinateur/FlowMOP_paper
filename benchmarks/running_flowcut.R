# Basic FlowCUT Batch Processing - Just paste and run!

# Load libraries
library(flowCore)
library(flowCut)

# Get all .fcs files in current directory
fcs_files <- list.files(".", pattern = "\\.fcs$", full.names = TRUE, ignore.case = TRUE)

if (length(fcs_files) == 0) {
  stop("No .fcs files found in current directory")
}

# Create output directory
output_dir <- file.path(getwd(), "flowcut_files")
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

cat("Found", length(fcs_files), ".fcs files. Processing...\n")
cat("Output directory:", output_dir, "\n")

# Process each file
for (i in seq_along(fcs_files)) {
  file_path <- fcs_files[i]
  file_name <- basename(file_path)
  file_base <- tools::file_path_sans_ext(file_name)
  
  cat("Processing", i, "of", length(fcs_files), ":", file_name, "...")
  
  # Read file
  flow_data <- read.FCS(file_path)
  
  # Run FlowCUT (minimal parameters, no plots)
  result <- flowCut(f = flow_data, Plot = "None", PrintToConsole = FALSE, Verbose = FALSE)
  
  # Save cleaned file to output directory
  output_name <- file.path(output_dir, paste0(file_base, "_flowcut.fcs"))
  write.FCS(result$frame, filename = output_name)
  
  cat(" Done\n")
}

cat("All files processed! Cleaned files saved in '_flowcut' directory.\n")
