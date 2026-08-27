# Figures 5 and 6: PBMC biological validation

This directory contains the reproducible event-level analysis for main Figures 5 and 6.
The inferential analysis uses one PBMC sample from each of eight independent
donors: 1B, 5B, 10A, 11B, 16A, 19A, 20A, and 22A.
Each selected repeat contains FSC-A and SSC-A and complete debris, doublet,
comparator, and manual data.

## Run

Install `flowkit`, `flowio`, `numpy`, `pandas`, `matplotlib`, `scipy`, and
`cairosvg`, then run from the repository root:

```bash
python figs_data/fig_5_data/generate_biological_validation_figure.py \
  --source-dir /path/to/flowmop_data/pbmc_biological_validation
```

On systems where Python is exposed as `python3`, substitute `python3` in the
same command. `--output-dir` and `--data-dir` may be used to redirect the
figure and table outputs.

## Analysis

The generator imports expert-defined FlowJo 10 gates with FlowKit. Because the
workspace contains three branches with duplicate display names, it creates a
temporary, one-sample FlowMOP workspace for each biological sample before
extracting event-level memberships. Gate geometry is hashed after removing
FlowJo IDs and display styling; the two singlet gates, debris gate, live-cell
gate, CD45 gate, all four CD19 × CD3 quadrants, and named NKT hierarchy must be
coordinate-identical across the FlowMOP, FlowCut, and PeacoQC branches for all
eight independent samples.

Manually cleaned, FlowCut, and PeacoQC events are mapped to the row-complete
FCS file by their six shared FSC/SSC pulse-geometry channels. FlowJo can
rewrite Time and compensated fluorescence values during manual export, so
those values are deliberately not used as identity keys. Every retained event
must have one exact, unique match and the complete mapping must be strictly
monotonic.

The biological endpoints are a standalone Live CD45+ reference, B cells
defined using the expert-defined Q1 (CD3−/CD19+) coordinates within Live cells,
T cells defined using the expert-defined Q3 (CD3+/CD19−) coordinates within Live cells, and NKT
cells defined as Live CD19− CD3+ CD56+ events. The lineage endpoints reuse the
expert coordinates without inheriting the CD45+ parent. The matched ungated inputs are:

- time: Expert singlet and debris masks, with no time mask;
- debris: Expert time and doublet masks, with no debris mask;
- doublet: Expert time and debris masks, with no doublet mask;
- all steps: no time, debris, or doublet preprocessing mask.

Every endpoint count is divided by its matched ungated-input count, with Raw
equal to exactly 1. Live CD45+, B, T, and NKT frequencies are expressed
relative to Live cells before Raw normalization. Figure 5B (frequencies) and Figure 5C
(counts) test all ten pairwise contrasts among Raw, Expert Manual,
FlowMOP, PeacoQC, and FlowCut, with Holm adjustment separately within each
endpoint and metric. The same three contrasts are tested separately for the
debris, doublet, and combined comparisons, with Holm adjustment within each
endpoint, metric, and gate group. All endpoints use `n = 8` and all Raw
denominators must be valid and nonzero.

Representative samples must have complete endpoint data and at least 5,000
live CD45+ reference events under every displayed workflow. Figure 5A uses
sample 19A, Figure 6 uses the selected representative from the eight-sample
inferential cohort, and Supplementary Figure S8 uses sample 19A for its debris
block.
Its doublet block uses the Figure 6 cohort representative.
Representative plots show retained events only and use
up to 100,000 events, fine-grid smoothed local-density estimates, sub-point
event marks, and the deep-blue-through-cyan, green, yellow, and red FlowJo
pseudocolor scale.

Counts and biological frequencies are each divided within sample by their
corresponding matched Raw value, so Raw is exactly 1 for both metrics. Before
normalization, all four frequencies are calculated relative to Live cells.

Figure 5A is a 4 × 5 grid of Time, Live CD45+, shared CD19 × CD3 B/T quadrants,
and NKT representatives, including a completely Raw column followed by the
four time workflows; Figure 5B contains the matched time-cleaning frequencies
and Figure 5C contains the matched time-cleaning counts for B, T, and NKT cells.
Main Figure 6A contains the combined Time + Debris + Doublet representative,
including Time, debris, doublet, Live CD45+, B/T, and NKT columns. Figure 6B
contains frequencies and Figure 6C contains counts; within each population,
Debris, Doublet, and Combined are displayed as three subcolumns containing Raw,
Expert Manual, and FlowMOP. Brackets and adjusted p values are drawn only for
significant pairwise comparisons. Supplementary Figure S8 contains only the
module-specific Debris and Doublet representatives. The rows
are Ungated input, Expert Manual, and FlowMOP.
Doublet rows use the expert-defined
FSC-H × FSC-W and SSC-H × SSC-W axes and rectangles; the FlowMOP-retained
events are projected onto those same axes. Every row also shows Live CD45+,
the shared B/T quadrant plot, and NKT after its corresponding mask. Axes are
asserted identical within every representative comparison. Representative
frequencies use semi-transparent white labels; B and T frequencies are placed
in Q1 and Q3. Figure 6 statistical panels display all three pairwise brackets
for Raw, Expert Manual, and FlowMOP within every Debris, Doublet, and Combined
subcolumn; bracket labels use `p` for brevity. Figure 5 continues to omit
Raw-comparison brackets to preserve legibility. Direct workflow comparisons
use paired t-tests; Raw-normalized workflow values are tested against 1 with
one-sample t-tests, which are equivalent to paired comparisons with matched
Raw. `biological_validation_cleaning_retention.csv` records event-level
retention before any biological endpoint gate. Figure 5A uses one
arrow-axis key per row, and the representative cleanup blocks use one per
relevant column.

## Outputs

- `../figure_5.svg` and `../figure_5.png`: time-cleaning Figure 5;
- `../figure_6.svg` and `../figure_6.png`: combined-cleaning main Figure 6;
- `../Supp_fig_8.svg` and `../Supp_fig_8.png`: module-specific supplementary validation;
- `biological_validation_endpoint_counts.csv`: plotted absolute counts, frequency denominators, and frequencies;
- `biological_validation_raw_normalized_ratios.csv`: all plotted raw-normalized counts and frequencies, their Raw denominators, and exclusions;
- `biological_validation_paired_tests.csv`: raw and Holm-adjusted count and frequency p-values;
- `biological_validation_gate_validation.csv`: gate-identity, population-count, FCS-count, and mask validations;
- `representative_sample_selection.csv`: eligibility, distance, rank, and endpoint vectors;
- `run_metadata.json`: input, software, sample, mask, and output summary.

The validation table records both stored FlowJo population counts and FlowKit
event-level recounts. Small differences can occur for polygon boundaries and
biexponential transforms; they are retained explicitly rather than silently
replacing either result. Exact invariants (sample sets, event reconstruction,
manual-export counts, module/comparator counts, and final-mask identity) abort
generation if violated.
