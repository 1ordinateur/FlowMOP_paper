#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(flowCore)
  library(PeacoQC)
  library(flowCut)
})

args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(3, 4)) {
  stop(paste(
    "Usage: analyse_fluorescence_instability.R",
    "INPUT_DIR FLOWMOP_DIR OUTPUT_DIR [BIN_SIZE]"
  ))
}

input_dir <- normalizePath(args[[1]], mustWork = TRUE)
flowmop_dir <- normalizePath(args[[2]], mustWork = TRUE)
output_dir <- args[[3]]
plot_dir <- file.path(output_dir, "sample_plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

bin_size <- if (length(args) == 4) as.integer(args[[4]]) else 1000L
if (!is.finite(bin_size) || bin_size < 100L) {
  stop("BIN_SIZE must be an integer of at least 100 events")
}
minimum_last_bin <- as.integer(bin_size %/% 2L)
asinh_cofactor <- 150
flagged_fraction <- 0.5
top_fraction <- 0.1
methods <- c("FlowMOP", "FlowCut", "PeacoQC")

input_files <- sort(list.files(
  input_dir,
  pattern = "^trim_.*\\.fcs$",
  full.names = TRUE,
  ignore.case = TRUE
))
if (length(input_files) == 0) {
  stop("No trim_*.fcs inputs found")
}

robust_z <- function(values) {
  center <- median(values, na.rm = TRUE)
  scale <- mad(values, center = center, constant = 1.4826, na.rm = TRUE)
  if (!is.finite(scale) || scale <= .Machine$double.eps) {
    scale <- IQR(values, na.rm = TRUE) / 1.349
  }
  if (!is.finite(scale) || scale <= .Machine$double.eps) {
    scale <- 1
  }
  (values - center) / scale
}

make_bins <- function(number_events) {
  bins <- ((seq_len(number_events) - 1L) %/% bin_size) + 1L
  last_bin <- max(bins)
  if (last_bin > 1L && sum(bins == last_bin) < minimum_last_bin) {
    bins[bins == last_bin] <- last_bin - 1L
  }
  bins
}

bin_summaries <- function(fluorescence, bins) {
  bin_ids <- sort(unique(bins))
  number_channels <- ncol(fluorescence)
  medians <- t(vapply(bin_ids, function(bin_id) {
    apply(fluorescence[bins == bin_id, , drop = FALSE], 2, median, na.rm = TRUE)
  }, numeric(number_channels)))
  upper_tails <- t(vapply(bin_ids, function(bin_id) {
    apply(
      fluorescence[bins == bin_id, , drop = FALSE],
      2,
      quantile,
      probs = 0.9,
      names = FALSE,
      na.rm = TRUE,
      type = 8
    )
  }, numeric(number_channels)))

  colnames(medians) <- paste0(colnames(fluorescence), "__median")
  colnames(upper_tails) <- paste0(colnames(fluorescence), "__q90")
  median_z <- apply(medians, 2, robust_z)
  upper_z <- apply(upper_tails, 2, robust_z)
  if (is.null(dim(median_z))) {
    median_z <- matrix(median_z, ncol = 1)
    upper_z <- matrix(upper_z, ncol = 1)
  }

  channel_divergence <- sqrt((median_z^2 + upper_z^2) / 2)
  colnames(channel_divergence) <- paste0(
    "channel_divergence__",
    colnames(fluorescence)
  )
  feature_z <- cbind(median_z, upper_z)
  colnames(feature_z) <- paste0("robust_z__", colnames(feature_z))

  cbind(data.frame(
    bin = bin_ids,
    bin_start_event = vapply(bin_ids, function(value) min(which(bins == value)), integer(1)),
    bin_end_event = vapply(bin_ids, function(value) max(which(bins == value)), integer(1)),
    bin_events = vapply(bin_ids, function(value) sum(bins == value), integer(1)),
    median_divergence = sqrt(rowMeans(median_z^2)),
    upper_tail_divergence = sqrt(rowMeans(upper_z^2)),
    combined_divergence = sqrt(rowMeans(cbind(median_z, upper_z)^2)),
    channels_with_divergence_ge_2 = rowSums(channel_divergence >= 2),
    stringsAsFactors = FALSE
  ), as.data.frame(channel_divergence, check.names = FALSE),
  as.data.frame(feature_z, check.names = FALSE))
}

circular_shift <- function(values, amount) {
  if (amount == 0L) {
    return(values)
  }
  c(tail(values, amount), head(values, -amount))
}

weighted_difference <- function(divergence, removed_fraction) {
  retained_fraction <- 1 - removed_fraction
  if (sum(removed_fraction) <= 0 || sum(retained_fraction) <= 0) {
    return(NA_real_)
  }
  weighted.mean(divergence, removed_fraction) -
    weighted.mean(divergence, retained_fraction)
}

method_bin_analysis <- function(sample_name, group_name, method_name, pass_mask,
                                bins, divergence_table) {
  removed <- !pass_mask
  removed_fraction <- as.numeric(tapply(removed, bins, mean))
  removed_events <- as.integer(tapply(removed, bins, sum))
  retained_events <- as.integer(tapply(pass_mask, bins, sum))
  divergence <- divergence_table$combined_divergence
  observed_delta <- weighted_difference(divergence, removed_fraction)

  circular_p <- NA_real_
  if (is.finite(observed_delta)) {
    shifts <- seq_len(length(removed_fraction) - 1L)
    null_deltas <- vapply(shifts, function(shift) {
      weighted_difference(divergence, circular_shift(removed_fraction, shift))
    }, numeric(1))
    circular_p <- (1 + sum(null_deltas >= observed_delta, na.rm = TRUE)) /
      (1 + sum(is.finite(null_deltas)))
  }

  top_cutoff <- quantile(
    divergence,
    probs = 1 - top_fraction,
    names = FALSE,
    type = 8
  )
  top_bins <- divergence >= top_cutoff
  removed_weight_in_top <- if (sum(removed_fraction) > 0) {
    sum(removed_fraction[top_bins]) / sum(removed_fraction)
  } else {
    NA_real_
  }

  weighted_removed_divergence <- if (sum(removed_fraction) > 0) {
    weighted.mean(divergence, removed_fraction)
  } else {
    NA_real_
  }
  weighted_retained_divergence <- if (sum(1 - removed_fraction) > 0) {
    weighted.mean(divergence, 1 - removed_fraction)
  } else {
    NA_real_
  }

  bin_result <- data.frame(
    sample = sample_name,
    group = group_name,
    method = method_name,
    divergence_table,
    removed_fraction = removed_fraction,
    removed_events = removed_events,
    retained_events = retained_events,
    flagged = removed_fraction >= flagged_fraction,
    top_divergence_decile = top_bins,
    stringsAsFactors = FALSE
  )

  summary_result <- data.frame(
    sample = sample_name,
    group = group_name,
    method = method_name,
    input_events = length(pass_mask),
    retained_events = sum(pass_mask),
    retention_percent = 100 * mean(pass_mask),
    bins = nrow(divergence_table),
    bins_with_any_removal = sum(removed_fraction > 0),
    majority_removed_bins = sum(removed_fraction >= flagged_fraction),
    weighted_removed_divergence = weighted_removed_divergence,
    weighted_retained_divergence = weighted_retained_divergence,
    divergence_delta = observed_delta,
    divergence_ratio = weighted_removed_divergence / weighted_retained_divergence,
    circular_shift_p = circular_p,
    removed_weight_in_top_divergence_decile = removed_weight_in_top,
    stringsAsFactors = FALSE
  )

  list(bins = bin_result, summary = summary_result)
}

plot_sample <- function(sample_name, divergence_table, sample_bins) {
  output_path <- file.path(
    plot_dir,
    paste0(tools::file_path_sans_ext(sample_name), "_fluorescence_instability.png")
  )
  png(output_path, width = 1800, height = 1200, res = 180)
  old_parameters <- par(no.readonly = TRUE)
  on.exit({
    par(old_parameters)
    dev.off()
  }, add = TRUE)
  par(mfrow = c(2, 1), mar = c(4.5, 4.8, 3, 1), oma = c(1, 0, 2, 0))

  plot(
    divergence_table$bin,
    divergence_table$combined_divergence,
    type = "l",
    lwd = 1.8,
    col = "#222222",
    xlab = "Acquisition-order bin",
    ylab = "Robust fluorescence divergence",
    main = "Fluorescence-only instability"
  )
  abline(
    h = quantile(divergence_table$combined_divergence, 0.9, type = 8),
    lty = 2,
    col = "#777777"
  )

  method_colours <- c(FlowMOP = "#4C72B0", FlowCut = "#55A868", PeacoQC = "#DD8452")
  wide <- sapply(methods, function(method_name) {
    sample_bins$removed_fraction[sample_bins$method == method_name]
  })
  matplot(
    divergence_table$bin,
    wide,
    type = "l",
    lty = 1,
    lwd = 1.8,
    col = method_colours[methods],
    xlab = "Acquisition-order bin",
    ylab = "Fraction removed",
    ylim = c(0, 1),
    main = "Method-specific removal"
  )
  abline(h = flagged_fraction, lty = 3, col = "#777777")
  legend(
    "topright",
    legend = methods,
    col = method_colours[methods],
    lty = 1,
    lwd = 2,
    bty = "n"
  )
  mtext(sample_name, outer = TRUE, cex = 1.15, font = 2)
}

all_bin_rows <- list()
all_summary_rows <- list()

for (file_index in seq_along(input_files)) {
  input_file <- input_files[[file_index]]
  sample_name <- basename(input_file)
  sample_stem <- tools::file_path_sans_ext(sample_name)
  group_name <- if (grepl("_NT", sample_name, fixed = TRUE)) "Non-tumour" else "Tumour"
  cat(sprintf("[%d/%d] %s\n", file_index, length(input_files), sample_name))

  ff <- read.FCS(input_file, transformation = FALSE, truncate_max_range = FALSE)
  raw_data <- exprs(ff)
  channel_names <- colnames(raw_data)
  fluorescence_channels <- which(!grepl("FSC|SSC|Time", channel_names, ignore.case = TRUE))
  fluorescence <- asinh(raw_data[, fluorescence_channels, drop = FALSE] / asinh_cofactor)
  bins <- make_bins(nrow(raw_data))
  divergence_table <- bin_summaries(fluorescence, bins)

  flowmop_path <- file.path(flowmop_dir, paste0("flowmop_", sample_name))
  flowmop_frame <- read.FCS(
    flowmop_path,
    transformation = FALSE,
    truncate_max_range = FALSE
  )
  if (nrow(flowmop_frame) != nrow(ff)) {
    stop(sprintf("FlowMOP event count mismatch for %s", sample_name))
  }
  normalized_names <- tolower(gsub("[^A-Za-z0-9]", "", colnames(flowmop_frame)))
  passed_time_index <- which(normalized_names == "passedtime")
  if (length(passed_time_index) != 1L) {
    stop(sprintf("Expected exactly one passed_time channel for %s", sample_name))
  }
  flowmop_pass <- exprs(flowmop_frame)[, passed_time_index] > 0.5

  flowcut_result <- flowCut::flowCut(
    f = ff,
    Plot = "None",
    PrintToConsole = FALSE,
    Verbose = FALSE
  )
  flowcut_pass <- rep(TRUE, nrow(ff))
  flowcut_pass[flowcut_result$ind] <- FALSE
  if (sum(flowcut_pass) != nrow(flowcut_result$frame)) {
    stop(sprintf("FlowCut mask mismatch for %s", sample_name))
  }

  peacoqc_result <- PeacoQC::PeacoQC(
    ff = ff,
    channels = fluorescence_channels,
    plot = FALSE,
    save_fcs = FALSE,
    report = FALSE,
    output_directory = tempdir()
  )
  peacoqc_pass <- as.logical(peacoqc_result$GoodCells)
  if (length(peacoqc_pass) != nrow(ff)) {
    stop(sprintf("PeacoQC mask mismatch for %s", sample_name))
  }

  masks <- list(
    FlowMOP = flowmop_pass,
    FlowCut = flowcut_pass,
    PeacoQC = peacoqc_pass
  )
  sample_bin_rows <- list()
  for (method_name in methods) {
    result <- method_bin_analysis(
      sample_name,
      group_name,
      method_name,
      masks[[method_name]],
      bins,
      divergence_table
    )
    all_bin_rows[[length(all_bin_rows) + 1L]] <- result$bins
    sample_bin_rows[[length(sample_bin_rows) + 1L]] <- result$bins
    all_summary_rows[[length(all_summary_rows) + 1L]] <- result$summary
  }
  plot_sample(sample_name, divergence_table, do.call(rbind, sample_bin_rows))
}

bin_results <- do.call(rbind, all_bin_rows)
sample_summary <- do.call(rbind, all_summary_rows)
write.csv(bin_results, file.path(output_dir, "bin_level_fluorescence_divergence.csv"), row.names = FALSE)
write.csv(sample_summary, file.path(output_dir, "method_sample_summary.csv"), row.names = FALSE)

call_pattern_rows <- list()
for (sample_name in unique(bin_results$sample)) {
  selected <- bin_results$sample == sample_name
  sample_data <- bin_results[selected, , drop = FALSE]
  reference_columns <- c(
    "sample",
    "group",
    "bin",
    "bin_start_event",
    "bin_end_event",
    "bin_events",
    "combined_divergence",
    "channels_with_divergence_ge_2",
    "top_divergence_decile",
    grep(
      "^(channel_divergence__|robust_z__)",
      colnames(sample_data),
      value = TRUE
    )
  )
  reference <- sample_data[
    sample_data$method == methods[[1]],
    reference_columns,
    drop = FALSE
  ]
  flags <- sapply(methods, function(method_name) {
    sample_data$flagged[sample_data$method == method_name]
  })
  colnames(flags) <- methods
  reference$call_pattern <- apply(flags, 1, function(row) {
    called <- methods[as.logical(row)]
    if (length(called) == 0) "None" else paste(called, collapse = "+")
  })
  reference$number_methods <- rowSums(flags)
  call_pattern_rows[[length(call_pattern_rows) + 1L]] <- reference
}
call_patterns <- do.call(rbind, call_pattern_rows)
write.csv(call_patterns, file.path(output_dir, "bin_call_patterns.csv"), row.names = FALSE)

top_aberrant_rows <- list()
for (sample_name in unique(call_patterns$sample)) {
  values <- call_patterns[call_patterns$sample == sample_name, , drop = FALSE]
  values <- values[order(values$combined_divergence, decreasing = TRUE), , drop = FALSE]
  values$divergence_rank <- seq_len(nrow(values))
  top_aberrant_rows[[length(top_aberrant_rows) + 1L]] <- head(values, 10)
}
write.csv(
  do.call(rbind, top_aberrant_rows),
  file.path(output_dir, "top_aberrant_bins.csv"),
  row.names = FALSE
)

pattern_summary_rows <- list()
for (group_name in c("Non-tumour", "Tumour", "All")) {
  selected_group <- if (group_name == "All") {
    rep(TRUE, nrow(call_patterns))
  } else {
    call_patterns$group == group_name
  }
  for (pattern_name in unique(call_patterns$call_pattern[selected_group])) {
    selected <- selected_group & call_patterns$call_pattern == pattern_name
    values <- call_patterns[selected, , drop = FALSE]
    pattern_summary_rows[[length(pattern_summary_rows) + 1L]] <- data.frame(
      group = group_name,
      call_pattern = pattern_name,
      bins = nrow(values),
      samples_represented = length(unique(values$sample)),
      median_divergence = median(values$combined_divergence),
      mean_divergence = mean(values$combined_divergence),
      top_divergence_decile_percent = 100 * mean(values$top_divergence_decile),
      stringsAsFactors = FALSE
    )
  }
}
pattern_summary <- do.call(rbind, pattern_summary_rows)
write.csv(pattern_summary, file.path(output_dir, "call_pattern_summary.csv"), row.names = FALSE)

cohort_rows <- list()
for (group_name in c("Non-tumour", "Tumour", "All")) {
  for (method_name in methods) {
    selected <- sample_summary$method == method_name
    if (group_name != "All") {
      selected <- selected & sample_summary$group == group_name
    }
    values <- sample_summary[selected, , drop = FALSE]
    cohort_rows[[length(cohort_rows) + 1L]] <- data.frame(
      group = group_name,
      method = method_name,
      samples = nrow(values),
      samples_with_removal = sum(is.finite(values$divergence_delta)),
      mean_retention_percent = mean(values$retention_percent),
      median_divergence_delta = median(values$divergence_delta, na.rm = TRUE),
      minimum_divergence_delta = min(values$divergence_delta, na.rm = TRUE),
      maximum_divergence_delta = max(values$divergence_delta, na.rm = TRUE),
      median_divergence_ratio = median(values$divergence_ratio, na.rm = TRUE),
      circular_p_le_0_05 = sum(values$circular_shift_p <= 0.05, na.rm = TRUE),
      median_removed_weight_in_top_decile = median(
        values$removed_weight_in_top_divergence_decile,
        na.rm = TRUE
      ),
      stringsAsFactors = FALSE
    )
  }
}
cohort_summary <- do.call(rbind, cohort_rows)
write.csv(cohort_summary, file.path(output_dir, "cohort_summary.csv"), row.names = FALSE)

parameters <- data.frame(
  parameter = c(
    "bin_size",
    "minimum_last_bin",
    "asinh_cofactor",
    "bin_flagged_removed_fraction",
    "top_divergence_fraction",
    "fluorescence_channels",
    "divergence_features",
    "primary_test"
  ),
  value = c(
    bin_size,
    minimum_last_bin,
    asinh_cofactor,
    flagged_fraction,
    top_fraction,
    "All non-Time/non-FSC/non-SSC channels",
    "Per-channel bin median and 90th percentile, robustly scaled within sample",
    "Exact one-sided circular shift of each method removal-fraction series"
  ),
  stringsAsFactors = FALSE
)
write.csv(parameters, file.path(output_dir, "analysis_parameters.csv"), row.names = FALSE)

cat("\nSAMPLE-METHOD SUMMARY\n")
print(sample_summary[, c(
  "sample",
  "group",
  "method",
  "retention_percent",
  "divergence_ratio",
  "divergence_delta",
  "circular_shift_p",
  "removed_weight_in_top_divergence_decile"
)], row.names = FALSE)
cat("\nCOHORT SUMMARY\n")
print(cohort_summary, row.names = FALSE)
cat(sprintf("\nOutputs written to %s\n", normalizePath(output_dir)))
