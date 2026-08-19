#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(flowCore))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop(
    "Usage: compare_biological_timegate_retention.R RAW_DIR FLOWMOP_DIR COMPARATOR_SUMMARY"
  )
}

raw_dir <- normalizePath(args[[1]], mustWork = TRUE)
flowmop_dir <- normalizePath(args[[2]], mustWork = TRUE)
comparator_summary <- normalizePath(args[[3]], mustWork = TRUE)

comparators <- read.csv(comparator_summary, stringsAsFactors = FALSE)
files <- sort(list.files(raw_dir, pattern = "\\.fcs$", ignore.case = TRUE))

flowmop_rows <- lapply(files, function(file_name) {
  path <- file.path(flowmop_dir, paste0("flowmop_", file_name))
  ff <- read.FCS(path, transformation = FALSE, truncate_max_range = FALSE)
  normalized_names <- tolower(gsub("[^A-Za-z0-9]", "", colnames(ff)))
  passed_time_index <- which(normalized_names == "passedtime")
  if (length(passed_time_index) != 1) {
    stop(sprintf("Expected one passed_time channel in %s", path))
  }
  retained <- exprs(ff)[, passed_time_index] > 0.5
  data.frame(
    file = file_name,
    method = "FlowMOP",
    input_events = nrow(ff),
    retained_events = sum(retained),
    removed_events = sum(!retained),
    retained_percent = 100 * mean(retained),
    stringsAsFactors = FALSE
  )
})
flowmop <- do.call(rbind, flowmop_rows)

comparison_columns <- c(
  "file",
  "method",
  "input_events",
  "retained_events",
  "removed_events",
  "retained_percent"
)
combined <- rbind(flowmop[, comparison_columns], comparators[, comparison_columns])

cat("METHOD SUMMARY\n")
print(aggregate(
  retained_percent ~ method,
  combined,
  function(values) c(
    mean = mean(values),
    median = median(values),
    min = min(values),
    max = max(values),
    sd = sd(values)
  )
))

wide <- reshape(
  combined[, c("file", "method", "retained_percent")],
  idvar = "file",
  timevar = "method",
  direction = "wide"
)
names(wide) <- sub("retained_percent\\.", "", names(wide))

cat("\nPAIRED DIFFERENCES (FlowMOP minus comparator, percentage points)\n")
for (method in c("PeacoQC", "FlowCut")) {
  differences <- wide$FlowMOP - wide[[method]]
  paired_test <- t.test(wide$FlowMOP, wide[[method]], paired = TRUE)
  cat(sprintf(
    paste0(
      "%s: mean=%.6f median=%.6f min=%.6f max=%.6f ",
      "more=%d equal=%d less=%d paired_t_p=%.12g\n"
    ),
    method,
    mean(differences),
    median(differences),
    min(differences),
    max(differences),
    sum(differences > 1e-9),
    sum(abs(differences) <= 1e-9),
    sum(differences < -1e-9),
    paired_test$p.value
  ))
}

cat("\nEVENT-WEIGHTED RETENTION\n")
weighted <- aggregate(
  cbind(input_events, retained_events) ~ method,
  combined,
  sum
)
weighted$retained_percent <- 100 * weighted$retained_events / weighted$input_events
print(weighted)

sample_number <- suppressWarnings(as.integer(sub("[A-Z].*", "", wide$file)))
wide <- wide[order(sample_number, wide$file), ]
cat("\nPER-FILE RETENTION (%)\n")
print(wide, row.names = FALSE)
