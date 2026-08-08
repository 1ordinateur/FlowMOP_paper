#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1 || length(args) > 2) {
  stop("Usage: install_full_timegating_r_dependencies.R R_LIBRARY [SOURCE_REPO]")
}

library_dir <- normalizePath(args[[1]], mustWork = FALSE)
dir.create(library_dir, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(R_LIBS_USER = library_dir)
.libPaths(c(library_dir, .libPaths()))
required_packages <- c("flowCore", "PeacoQC", "flowCut")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  if (length(args) == 2) {
    source_repo <- normalizePath(args[[2]], mustWork = TRUE)
    install.packages(
      missing_packages,
      lib = library_dir,
      repos = paste0("file://", source_repo),
      type = "source",
      dependencies = c("Depends", "Imports", "LinkingTo")
    )
  } else {
    options(
      repos = c(
        CRAN = "https://packagemanager.posit.co/cran/2024-04-30"
      )
    )
    if (!requireNamespace("BiocManager", quietly = TRUE)) {
      install.packages("BiocManager", lib = library_dir)
    }
    BiocManager::install(
      missing_packages,
      lib = library_dir,
      version = "3.18",
      ask = FALSE,
      update = FALSE,
      Ncpus = 1
    )
  }
}

remaining <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(remaining) > 0) {
  stop("Installation did not provide: ", paste(remaining, collapse = ", "))
}

version_lines <- vapply(
  required_packages,
  function(package) paste0(package, "=", as.character(packageVersion(package))),
  character(1)
)
cat("R_DEPENDENCIES_OK ", paste(version_lines, collapse = " "), "\n", sep = "")
