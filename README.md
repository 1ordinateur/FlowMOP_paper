# FlowMOP Paper

Code and analysis for the FlowMOP flow cytometry quality control paper. This repository contains algorithm validation, benchmarking scripts, and figure generation code.

## Repository Structure

```
├── algo_validation/     # Synthetic data generation and algorithm validation
├── benchmarks/          # Benchmarking against FlowAI, FlowCut, PeacoQC
├── figs_data/           # Figure generation and statistical analysis
└── utils/               # Utility scripts for data processing
```

## Installation

### Python Dependencies

```bash
pip install numpy pandas dask[dataframe] scipy statsmodels matplotlib seaborn readfcs fcsparser flowkit
```

### R Dependencies (for benchmark comparisons)

```R
install.packages("BiocManager")
BiocManager::install(c("flowCore", "PeacoQC", "flowAI", "flowCut"))
```

### System Requirements

- Python 3.8+
- R 4.0+ (for benchmark scripts)
- GNU Parallel (for batch processing)

## Usage

### Creating Synthetic Mixed FCS Files

Create mixed files with specific proportions:

```bash
python algo_validation/time_gate_executables/concatenate_fcs_cli.py \
    --specific-files file1.fcs file2.fcs \
    --specific-proportions 0.7 0.3 \
    --output-file mixed.fcs \
    --enable-mixing
```

Generate random combinations for validation:

```bash
python algo_validation/time_gate_executables/generate_batch_mixtures.py \
    --input-dir ./raw_fcs \
    --output-dir ./mixed_output \
    --num-combinations 50 \
    --files-per-combo 2 \
    --seed 42
```

### Running Benchmarks

**FlowMOP** (see [FlowMOP repository](https://github.com/1ordinateur/FlowMOP)):

```bash
python flowmop_exec.py input.fcs --output-dir ./output
```

**PeacoQC:**

```bash
Rscript benchmarks/run_peacoqc.R
```

**Parallel processing:**

```bash
bash benchmarks/parallel_flowai.sh input_dir output_dir 4
bash benchmarks/parallel_flowcut.sh input_dir output_dir 4
bash benchmarks/parallel_peacoqc.sh input_dir output_dir 4
```

### Utility Scripts

Parse training logs:

```bash
python utils/parse_logs.py --root-directory ./experiments --output-file results.csv
```

Benchmark file I/O performance:

```bash
python utils/simple_read_benchmark.py file.parquet
```

## Algorithm Validation

The `algo_validation/time_gate_disturbance/` module provides tools for creating synthetic flow cytometry files with known time-gate disturbances:

- **concatenators.py**: Concatenate multiple FCS files with specified proportions
- **sampling.py**: Sampling strategies for file selection
- **adjusters.py**: Time channel adjustment during concatenation
- **validation.py**: Input validation utilities

## Figure Generation

Jupyter notebooks in `figs_data/` reproduce the paper figures:

- `fig_2_data/`: Time gate, debris, and doublet detection analysis
- `fig_3_data/`: Comparative analysis with statistical tests
- `fig_4_data/`: Cleaning quantification results

## HPC Support

PBS job scripts for GADI cluster are in `benchmarks/clean_dataset_gadi/`:

- `flowmop_worker.pbs`: Main worker script with job chaining
- `flowmop_autosubmit_master.sh`: Master submission script
- `flowmop_monitor.sh`: Job monitoring utility

## Related

- [FlowMOP](https://github.com/1ordinateur/FlowMOP) - The main FlowMOP tool

## License

MIT License - see [LICENSE](LICENSE)
