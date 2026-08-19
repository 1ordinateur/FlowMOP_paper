#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(flowCore)
  library(PeacoQC)
  library(flowCut)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: run_default_biological_qc.R INPUT_DIR OUTPUT_DIR")
}

input_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
peacoqc_dir <- file.path(output_dir, "peacoqc")
flowcut_dir <- file.path(output_dir, "flowcut")
summary_path <- file.path(output_dir, "default_qc_summary.csv")

dir.create(peacoqc_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(flowcut_dir, recursive = TRUE, showWarnings = FALSE)

input_files <- sort(list.files(
  input_dir,
  pattern = "\\.fcs$",
  full.names = TRUE,
  ignore.case = TRUE
))
if (length(input_files) == 0) {
  stop("No FCS files found directly within INPUT_DIR")
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

summary_rows <- list()

record_result <- function(
  file_name,
  method,
  status,
  n_input,
  n_retained,
  elapsed_sec,
  output_file,
  message = "",
  events_per_bin = NA_integer_,
  nr_bins = NA_integer_,
  worst_channel = NA_character_
) {
  n_removed <- if (is.na(n_retained)) NA_integer_ else n_input - n_retained
  retained_pct <- if (is.na(n_retained)) NA_real_ else 100 * n_retained / n_input
  summary_rows[[length(summary_rows) + 1]] <<- data.frame(
    file = file_name,
    method = method,
    status = status,
    input_events = n_input,
    retained_events = n_retained,
    removed_events = n_removed,
    retained_percent = retained_pct,
    elapsed_seconds = elapsed_sec,
    events_per_bin = events_per_bin,
    number_of_bins = nr_bins,
    worst_channel = worst_channel,
    output_file = output_file,
    message = message,
    stringsAsFactors = FALSE
  )
  write.csv(do.call(rbind, summary_rows), summary_path, row.names = FALSE)
}

cat(sprintf("Found %d FCS files in %s\n", length(input_files), input_dir))
cat(sprintf("PeacoQC %s; flowCut %s; flowCore %s\n",
  as.character(packageVersion("PeacoQC")),
  as.character(packageVersion("flowCut")),
  as.character(packageVersion("flowCore"))
))

for (file_index in seq_along(input_files)) {
  input_file <- input_files[[file_index]]
  file_name <- basename(input_file)
  file_stem <- tools::file_path_sans_ext(file_name)
  cat(sprintf("[%d/%d] Reading %s\n", file_index, length(input_files), file_name))

  ff <- tryCatch(
    read.FCS(input_file, transformation = FALSE, truncate_max_range = FALSE),
    error = function(error) error
  )
  if (inherits(ff, "error")) {
    for (method in c("PeacoQC", "FlowCut")) {
      record_result(
        file_name,
        method,
        "read_error",
        NA_integer_,
        NA_integer_,
        NA_real_,
        "",
        conditionMessage(ff)
      )
    }
    next
  }

  n_input <- nrow(ff)
  channel_names <- colnames(ff)
  qc_channels <- which(!grepl("FSC|SSC|Time", channel_names, ignore.case = TRUE))
  if (length(qc_channels) == 0) {
    stop(sprintf("No fluorescence QC channels found in %s", file_name))
  }

  peacoqc_output <- file.path(peacoqc_dir, paste0(file_stem, "_peacoqc.fcs"))
  cat(sprintf("[%d/%d] PeacoQC %s\n", file_index, length(input_files), file_name))
  peacoqc_start <- proc.time()[["elapsed"]]
  peacoqc_result <- tryCatch(
    PeacoQC::PeacoQC(
      ff = ff,
      channels = qc_channels,
      plot = FALSE,
      save_fcs = FALSE,
      report = FALSE,
      output_directory = tempdir()
    ),
    error = function(error) error
  )
  peacoqc_elapsed <- proc.time()[["elapsed"]] - peacoqc_start
  if (inherits(peacoqc_result, "error")) {
    record_result(
      file_name,
      "PeacoQC",
      "error",
      n_input,
      NA_integer_,
      peacoqc_elapsed,
      "",
      conditionMessage(peacoqc_result)
    )
  } else {
    peacoqc_retained <- tryCatch(
      extract_good_cells(peacoqc_result),
      error = function(error) error
    )
    if (inherits(peacoqc_retained, "error") || length(peacoqc_retained) != n_input) {
      message_text <- if (inherits(peacoqc_retained, "error")) {
        conditionMessage(peacoqc_retained)
      } else {
        sprintf("GoodCells length %d did not match %d input events", length(peacoqc_retained), n_input)
      }
      record_result(
        file_name,
        "PeacoQC",
        "error",
        n_input,
        NA_integer_,
        peacoqc_elapsed,
        "",
        message_text
      )
    } else {
      write.FCS(ff[peacoqc_retained, ], peacoqc_output)
      record_result(
        file_name,
        "PeacoQC",
        "ok",
        n_input,
        sum(peacoqc_retained),
        peacoqc_elapsed,
        peacoqc_output,
        events_per_bin = suppressWarnings(as.integer(peacoqc_result$EventsPerBin)),
        nr_bins = suppressWarnings(as.integer(peacoqc_result$nr_bins))
      )
    }
  }

  flowcut_output <- file.path(flowcut_dir, paste0(file_stem, "_flowcut.fcs"))
  cat(sprintf("[%d/%d] FlowCut %s\n", file_index, length(input_files), file_name))
  flowcut_start <- proc.time()[["elapsed"]]
  flowcut_result <- tryCatch(
    flowCut::flowCut(
      f = ff,
      Plot = "None",
      PrintToConsole = FALSE,
      Verbose = FALSE
    ),
    error = function(error) error
  )
  flowcut_elapsed <- proc.time()[["elapsed"]] - flowcut_start
  if (inherits(flowcut_result, "error")) {
    record_result(
      file_name,
      "FlowCut",
      "error",
      n_input,
      NA_integer_,
      flowcut_elapsed,
      "",
      conditionMessage(flowcut_result)
    )
  } else {
    write.FCS(flowcut_result$frame, flowcut_output)
    record_result(
      file_name,
      "FlowCut",
      "ok",
      n_input,
      nrow(flowcut_result$frame),
      flowcut_elapsed,
      flowcut_output,
      worst_channel = if (is.null(flowcut_result$worstChan)) {
        NA_character_
      } else {
        as.character(flowcut_result$worstChan[[1]])
      }
    )
  }
}

cat(sprintf("Completed. Summary: %s\n", summary_path))
