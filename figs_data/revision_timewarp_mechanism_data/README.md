# Figure S4 mechanism-benchmark data

These files are the canonical source for Figure S4. They combine the corrected
PeacoQC and FlowCut rows from the original 30-file matched Time-warp benchmark
with the final FlowMOP rerun performed using MAD factor 5 and smoothing factors
`0.01,0.05`.

The FlowMOP rerun was Gadi job `175818850.gadi-pbs`. It completed all 30 source
files and all three matched variants (raw, source-linked Time warp, and random
Time warp), producing 90 successful FlowMOP rows. FlowMOP's retained events and
all reported scores were exactly identical across the three variants for every
file. The 180 competitor rows were not rerun or otherwise changed during this
update.

- `results.csv`: combined absolute results for all three algorithms.
- `results_with_raw_delta.csv`: canonical plotting input with matched-raw
  changes.
- `summary.csv`: mean absolute results by algorithm and variant.
- `flowmop_invariance_by_file.csv`: exact per-file invariance checks.
- `run_metadata.json`: settings and provenance summary.
- `parallel_joblog.tsv` and `gadi_job_175818850.log`: execution records.

The `input_fcs` field is normalized to a portable path of the form
`inputs/<file>_<variant>.fcs`; the large intermediate FCS files are not included
in the repository.

Regenerate the figure from the repository root with:

```bash
python3 benchmarks/plot_rate_density_mechanism.py
```
