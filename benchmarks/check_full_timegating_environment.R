#!/usr/bin/env Rscript

required_packages <- c("flowCore", "PeacoQC", "flowCut")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop("Missing R packages: ", paste(missing_packages, collapse = ", "))
}

cat("R_OK\n")
