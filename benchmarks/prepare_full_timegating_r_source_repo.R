#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: prepare_full_timegating_r_source_repo.R SOURCE_REPO")
}

source_repo <- normalizePath(args[[1]], mustWork = FALSE)
dir.create(source_repo, recursive = TRUE, showWarnings = FALSE)

repositories <- c(
  BioCsoft = "https://bioconductor.org/packages/3.18/bioc",
  BioCann = "https://bioconductor.org/packages/3.18/data/annotation",
  BioCexp = "https://bioconductor.org/packages/3.18/data/experiment",
  BioCworkflows = "https://bioconductor.org/packages/3.18/workflows",
  CRAN = "https://packagemanager.posit.co/cran/2024-04-30"
)
required_packages <- c("flowCore", "PeacoQC", "flowCut")
dependency_fields <- c("Depends", "Imports", "LinkingTo")

available <- available.packages(
  contriburl = contrib.url(repositories, type = "source"),
  type = "source"
)
missing_required <- setdiff(required_packages, rownames(available))
if (length(missing_required) > 0) {
  stop(
    "Required packages absent from the Bioconductor 3.18/CRAN indexes: ",
    paste(missing_required, collapse = ", ")
  )
}

dependency_map <- tools::package_dependencies(
  required_packages,
  db = available,
  which = dependency_fields,
  recursive = TRUE
)
packages <- unique(c(required_packages, unlist(dependency_map, use.names = FALSE)))
packages <- intersect(packages, rownames(available))

downloaded <- download.packages(
  packages,
  destdir = source_repo,
  available = available,
  repos = repositories,
  type = "source"
)
if (nrow(downloaded) != length(packages)) {
  downloaded_names <- downloaded[, "Package"]
  stop(
    "Failed to download: ",
    paste(setdiff(packages, downloaded_names), collapse = ", ")
  )
}

tools::write_PACKAGES(source_repo, type = "source", latestOnly = TRUE)
writeLines(
  c(
    paste0("bioconductor_release=3.18"),
    paste0("package_count=", length(packages)),
    paste0("prepared_at=", format(Sys.time(), tz = "UTC", usetz = TRUE)),
    paste0("required=", paste(required_packages, collapse = ","))
  ),
  file.path(source_repo, "SOURCE_REPO_METADATA.txt")
)
cat("R_SOURCE_REPO_OK packages=", length(packages), "\n", sep = "")
