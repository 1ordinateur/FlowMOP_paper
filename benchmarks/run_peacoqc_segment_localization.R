#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(flowCore)
  library(PeacoQC)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop(
    "Usage: run_peacoqc_segment_localization.R INPUT_FCS EVENTS_PER_BIN_OR_AUTO TARGET_IDS OUTPUT_DIR"
  )
}

input_fcs <- args[[1]]
setting <- args[[2]]
target_ids <- as.integer(strsplit(args[[3]], ",", fixed = TRUE)[[1]])
output_dir <- args[[4]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

sample_col_index <- function(names) {
  normalized <- tolower(gsub("[^A-Za-z0-9]", "", names))
  exact <- which(normalized == "sampleidint")
  if (length(exact) > 0) {
    return(exact[[1]])
  }
  candidates <- which(grepl("sample", normalized) & grepl("id", normalized))
  if (length(candidates) > 0) {
    return(candidates[[1]])
  }
  stop("SampleIDInt/source label channel not found")
}

safe_quantile <- function(values, probability) {
  values <- values[is.finite(values)]
  if (length(values) == 0) {
    return(NA_real_)
  }
  as.numeric(stats::quantile(values, probability, names = FALSE, type = 8))
}

transition_distances <- function(source_ids) {
  event_count <- length(source_ids)
  transitions <- which(source_ids[-1] != source_ids[-event_count])
  if (length(transitions) == 0) {
    return(list(indices = integer(0), distance = rep(NA_real_, event_count)))
  }
  event_indices <- seq_len(event_count)
  distance <- rep(Inf, event_count)
  for (transition in transitions) {
    distance <- pmin(
      distance,
      pmin(abs(event_indices - transition), abs(event_indices - (transition + 1L)))
    )
  }
  list(indices = transitions, distance = distance)
}

boundary_spillover <- function(removed_mask, target_mask, transitions) {
  spillovers <- integer(0)
  event_count <- length(removed_mask)
  for (transition in transitions) {
    left <- transition
    if (target_mask[[left]]) {
      count <- 0L
      while (left >= 1L && target_mask[[left]] && removed_mask[[left]]) {
        count <- count + 1L
        left <- left - 1L
      }
      spillovers <- c(spillovers, count)
    }
    right <- transition + 1L
    if (right <= event_count && target_mask[[right]]) {
      count <- 0L
      while (right <= event_count && target_mask[[right]] && removed_mask[[right]]) {
        count <- count + 1L
        right <- right + 1L
      }
      spillovers <- c(spillovers, count)
    }
  }
  if (length(spillovers) == 0) {
    return(c(max = NA_real_, total = NA_real_))
  }
  c(max = max(spillovers), total = sum(spillovers))
}

ff <- read.FCS(input_fcs, transformation = FALSE)
channel_names <- colnames(ff)
sample_index <- sample_col_index(channel_names)
source_ids <- as.integer(round(exprs(ff)[, sample_index]))
channels <- which(!grepl("FSC|SSC|Time|Sample", channel_names, ignore.case = TRUE))

peacoqc_args <- list(
  ff,
  channels = channels,
  determine_good_cells = "all",
  plot = FALSE,
  save_fcs = FALSE,
  report = FALSE,
  output_directory = tempdir()
)
if (tolower(setting) != "auto") {
  events_per_bin <- suppressWarnings(as.integer(setting))
  if (is.na(events_per_bin) || events_per_bin < 1) {
    stop("EVENTS_PER_BIN must be 'auto' or a positive integer")
  }
  peacoqc_args$events_per_bin <- events_per_bin
}
result <- do.call(PeacoQC::PeacoQC, peacoqc_args)

good_cells <- as.logical(result$GoodCells)
if (length(good_cells) != length(source_ids)) {
  stop("PeacoQC GoodCells mask does not match the input event count")
}
events_per_bin_used <- as.integer(result$EventsPerBin)
if (is.na(events_per_bin_used) || events_per_bin_used < 1) {
  stop("PeacoQC did not report a valid EventsPerBin value")
}

removed_mask <- !good_cells
target_mask <- source_ids %in% target_ids
nontarget_mask <- !target_mask
retained_count <- sum(good_cells)
removed_count <- sum(removed_mask)
retained_target_count <- sum(good_cells & target_mask)
removed_target_count <- sum(removed_mask & target_mask)
removed_nontarget_count <- sum(removed_mask & nontarget_mask)
target_count <- sum(target_mask)
nontarget_count <- sum(nontarget_mask)

distances <- transition_distances(source_ids)
removed_target_distances <- distances$distance[removed_mask & target_mask]
spillover <- boundary_spillover(removed_mask, target_mask, distances$indices)

run_summary <- data.frame(
  input_fcs = basename(input_fcs),
  requested_setting = setting,
  events = length(source_ids),
  events_per_bin_used = events_per_bin_used,
  nr_bins = as.integer(result$nr_bins),
  source_transition_count = length(distances$indices),
  target_source_ids = paste(target_ids, collapse = ","),
  retained_count = retained_count,
  removed_count = removed_count,
  retained_fraction = retained_count / length(source_ids),
  removed_fraction = removed_count / length(source_ids),
  retained_target_purity = if (retained_count > 0) retained_target_count / retained_count else NA_real_,
  removed_nontarget_purity = if (removed_count > 0) removed_nontarget_count / removed_count else NA_real_,
  target_retention = if (target_count > 0) retained_target_count / target_count else NA_real_,
  target_removal_fraction = if (target_count > 0) removed_target_count / target_count else NA_real_,
  nontarget_recall = if (nontarget_count > 0) removed_nontarget_count / nontarget_count else NA_real_,
  removed_target_count = removed_target_count,
  removed_nontarget_count = removed_nontarget_count,
  removed_target_distance_median = safe_quantile(removed_target_distances, 0.5),
  removed_target_distance_p95 = safe_quantile(removed_target_distances, 0.95),
  removed_target_distance_max = safe_quantile(removed_target_distances, 1.0),
  removed_target_beyond_one_bin_fraction = if (length(removed_target_distances) > 0) {
    mean(removed_target_distances > events_per_bin_used)
  } else {
    NA_real_
  },
  removed_target_beyond_two_bins_fraction = if (length(removed_target_distances) > 0) {
    mean(removed_target_distances > 2 * events_per_bin_used)
  } else {
    NA_real_
  },
  boundary_target_spillover_max = unname(spillover[["max"]]),
  boundary_target_spillover_total = unname(spillover[["total"]]),
  stringsAsFactors = FALSE
)
write.csv(run_summary, file.path(output_dir, "run_summary.csv"), row.names = FALSE)

# Run-length encoding split whenever either retention state or source identity changes.
keys <- paste(as.integer(good_cells), source_ids, sep = ":")
runs <- rle(keys)
run_ends <- cumsum(runs$lengths)
run_starts <- c(1L, head(run_ends, -1L) + 1L)
parts <- strsplit(runs$values, ":", fixed = TRUE)
mask_rle <- data.frame(
  start_event = run_starts,
  end_event = run_ends,
  events = runs$lengths,
  retained = as.integer(vapply(parts, `[[`, character(1), 1L)),
  source_id = as.integer(vapply(parts, `[[`, character(1), 2L))
)
write.csv(mask_rle, file.path(output_dir, "mask_rle.csv"), row.names = FALSE)
