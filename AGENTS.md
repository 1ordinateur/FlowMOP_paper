# Repository guidance

## Gadi submissions

- `expresssr` is a specific Gadi PBS queue and is distinct from `express`.
- When a request specifies `expresssr`, use the literal PBS queue setting `#PBS -q expresssr` (or `qsub -q expresssr`). Do not reinterpret it as `express`.
