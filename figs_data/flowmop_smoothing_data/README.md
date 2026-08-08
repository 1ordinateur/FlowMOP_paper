# Full-dataset FlowMOP smoothing analysis

These files contain the final MAD-5 smoothing analysis used for Supplementary
Table S4 and selection of the `0.01,0.05` default. The analysis comprises 19
smoothing settings evaluated on each of the 173 primary non-tied synthetic
time-gating inputs. Source labels were excluded from quality-control inputs and
used only for scoring.

- `primary_results_excluding_ties.csv`: all 3,287 per-file results.
- `means_by_benchmark_group.csv`: means for the six equally weighted benchmark
  groups defined by dataset, synthetic bin size, and mixture method.
- `tradeoff_summary.csv`: macro- and file-weighted summaries, trade-off metrics,
  Pareto status, and equal-weight ranks for all settings.
- `completion_report.json`: settings, sample counts, selection rule, and final
  selected results.

The no-smoothing control has the highest equal-weight balanced mean. Among the
smoothed settings, `0.01,0.05` has the highest equal-weight balanced mean and is
therefore the selected default.
