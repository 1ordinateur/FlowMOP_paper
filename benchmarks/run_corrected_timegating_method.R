#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(flowCore)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop(
    "Usage: run_corrected_timegating_method.R ALGORITHM INPUT_FCS TARGET_IDS OUTPUT_CSV"
  )
}

algorithm <- tolower(args[[1]])
input_fcs <- args[[2]]
target_ids <- as.integer(strsplit(args[[3]], ",", fixed = TRUE)[[1]])
output_csv <- args[[4]]

if (!algorithm %in% c("flowmop", "peacoqc", "flowcut")) {
  stop("ALGORITHM must be flowmop, peacoqc, or flowcut")
}
if (any(is.na(target_ids)) || length(target_ids) < 1) {
  stop("TARGET_IDS must contain at least one integer source ID")
}
dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)

normalize_channel_name <- function(value) {
  tolower(gsub("[^A-Za-z0-9]", "", value))
}

sample_col_index <- function(names) {
  normalized <- normalize_channel_name(names)
  exact <- which(normalized == "sampleidint")
  if (length(exact) != 1) {
    stop(
      sprintf(
        "Expected exactly one SampleIDInt source-label channel; found %d in: %s",
        length(exact), paste(names, collapse = " | ")
      )
    )
  }
  exact[[1]]
}

corrected_qc_channels <- function(names, sample_index) {
  normalized <- normalize_channel_name(names)
  structural <- grepl("FSC|SSC|Time", names, ignore.case = TRUE) |
    startsWith(normalized, "passed")
  channels <- setdiff(which(!structural), sample_index)
  selected_normalized <- normalize_channel_name(names[channels])
  if (sample_index %in% channels || any(selected_normalized == "sampleidint")) {
    stop("Data-leakage guard failed: SampleIDInt remains in the QC channels")
  }
  if (length(channels) < 1) {
    stop("No fluorescence QC channels remain after structural/source exclusions")
  }
  channels
}

package_version_or_na <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    return(NA_character_)
  }
  as.character(utils::packageVersion(package))
}

extract_good_cells <- function(result) {
  candidates <- list(
    result$GoodCells,
    result$goodCells,
    result$good_cells,
    result$goodcells,
    result$PeacoQC_result$GoodCells,
    result$PeacoQC_result$goodCells
  )
  for (candidate in candidates) {
    if (!is.null(candidate)) {
      return(as.logical(candidate))
    }
  }
  stop("PeacoQC did not return a GoodCells mask")
}

ff <- read.FCS(input_fcs, transformation = FALSE, truncate_max_range = FALSE)
channel_names <- colnames(ff)
sample_index <- sample_col_index(channel_names)
source_ids <- as.integer(round(exprs(ff)[, sample_index]))
qc_channels <- corrected_qc_channels(channel_names, sample_index)

events_per_bin_used <- NA_integer_
nr_bins <- NA_integer_
flowcut_worst_channel <- NA_character_
elapsed <- NA_real_

if (algorithm == "flowmop") {
  passed_candidates <- which(
    normalize_channel_name(channel_names) %in% c("passedtime", "passedtimegate")
  )
  if (length(passed_candidates) != 1) {
    stop(
      sprintf(
        "Expected exactly one passed_time channel in FlowMOP output; found %d",
        length(passed_candidates)
      )
    )
  }
  retained <- exprs(ff)[, passed_candidates[[1]]] > 0.5
} else if (algorithm == "peacoqc") {
  suppressPackageStartupMessages(library(PeacoQC))
  timing <- system.time({
    result <- PeacoQC::PeacoQC(
      ff,
      channels = qc_channels,
      determine_good_cells = "all",
      plot = FALSE,
      save_fcs = FALSE,
      report = FALSE,
      output_directory = tempdir()
    )
  })
  elapsed <- unname(timing[["elapsed"]])
  retained <- extract_good_cells(result)
  if (length(retained) != length(source_ids)) {
    stop("PeacoQC GoodCells mask length does not match the input event count")
  }
  events_per_bin_used <- suppressWarnings(as.integer(result$EventsPerBin))
  nr_bins <- suppressWarnings(as.integer(result$nr_bins))
} else {
  suppressPackageStartupMessages(library(flowCut))
  timing <- system.time({
    result <- flowCut::flowCut(
      f = ff,
      Channels = qc_channels,
      Plot = "None",
      PrintToConsole = FALSE,
      Verbose = FALSE
    )
  })
  elapsed <- unname(timing[["elapsed"]])
  retained_frame <- result$frame
  retained_names <- colnames(retained_frame)
  retained_sample_index <- sample_col_index(retained_names)
  retained_source_ids <- as.integer(
    round(exprs(retained_frame)[, retained_sample_index])
  )
  original_source_counts <- table(factor(source_ids, levels = sort(unique(source_ids))))
  retained_source_counts <- table(
    factor(retained_source_ids, levels = sort(unique(source_ids)))
  )
  removed_source_counts <- original_source_counts - retained_source_counts
  if (any(removed_source_counts < 0)) {
    stop("FlowCut retained source counts exceed the original source counts")
  }
  retained_target_count <- sum(
    retained_source_counts[names(retained_source_counts) %in% as.character(target_ids)]
  )
  removed_target_count <- sum(
    removed_source_counts[names(removed_source_counts) %in% as.character(target_ids)]
  )
  retained_count <- nrow(retained_frame)
  removed_count <- nrow(ff) - retained_count
  target_count <- sum(source_ids %in% target_ids)
  nontarget_count <- length(source_ids) - target_count
  removed_nontarget_count <- removed_count - removed_target_count
  flowcut_worst_channel <- if (!is.null(result$worstChan)) {
    paste(result$worstChan, collapse = "|")
  } else {
    NA_character_
  }
}

if (algorithm != "flowcut") {
  retained <- as.logical(retained)
  if (length(retained) != length(source_ids)) {
    stop("Retention mask length does not match the source-label vector")
  }
  target <- source_ids %in% target_ids
  removed <- !retained
  retained_count <- sum(retained)
  removed_count <- sum(removed)
  target_count <- sum(target)
  nontarget_count <- sum(!target)
  retained_target_count <- sum(retained & target)
  removed_target_count <- sum(removed & target)
  removed_nontarget_count <- sum(removed & !target)
}

retained_target_purity <- if (retained_count > 0) {
  retained_target_count / retained_count
} else {
  NA_real_
}
removed_nontarget_purity <- if (removed_count > 0) {
  removed_nontarget_count / removed_count
} else {
  NA_real_
}
target_retention <- if (target_count > 0) {
  retained_target_count / target_count
} else {
  NA_real_
}
nontarget_recall <- if (nontarget_count > 0) {
  removed_nontarget_count / nontarget_count
} else {
  NA_real_
}

summary <- data.frame(
  algorithm = algorithm,
  input_fcs = basename(input_fcs),
  events = length(source_ids),
  target_source_ids = paste(target_ids, collapse = ","),
  source_ids_present = paste(sort(unique(source_ids)), collapse = ","),
  sample_channel_index = sample_index,
  sample_channel_name = channel_names[[sample_index]],
  sample_channel_used_for_qc = sample_index %in% qc_channels,
  qc_channel_count = length(qc_channels),
  qc_channel_indices = paste(qc_channels, collapse = ","),
  qc_channel_names = paste(channel_names[qc_channels], collapse = "|"),
  retained_count = retained_count,
  removed_count = removed_count,
  retained_fraction = retained_count / length(source_ids),
  removed_fraction = removed_count / length(source_ids),
  retained_target_count = retained_target_count,
  removed_target_count = removed_target_count,
  removed_nontarget_count = removed_nontarget_count,
  retained_target_purity = retained_target_purity,
  removed_nontarget_purity = removed_nontarget_purity,
  target_retention = target_retention,
  nontarget_recall = nontarget_recall,
  events_per_bin_used = events_per_bin_used,
  nr_bins = nr_bins,
  flowcut_worst_channel = flowcut_worst_channel,
  elapsed_seconds = elapsed,
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  flowcore_version = package_version_or_na("flowCore"),
  peacoqc_version = package_version_or_na("PeacoQC"),
  flowcut_version = package_version_or_na("flowCut"),
  stringsAsFactors = FALSE
)

if (isTRUE(summary$sample_channel_used_for_qc)) {
  stop("Data-leakage guard failed in final run summary")
}
write.csv(summary, output_csv, row.names = FALSE)
