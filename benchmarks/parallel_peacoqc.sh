#!/bin/bash
# Run PeacoQC on multiple FCS files in parallel

set -e

INPUT_DIR="$1"
OUTPUT_DIR="$2"
MAX_PARALLEL="${3:-4}"

if [[ -z "$INPUT_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 <input_dir> <output_dir> [max_parallel]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Find all FCS files
FILES=( $(find "$INPUT_DIR" -type f -iname "*.fcs") )

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No FCS files found"
    exit 0
fi

echo "Processing ${#FILES[@]} files with PeacoQC"

# Function to run PeacoQC
run_peacoqc() {
    local input_file="$1"
    local output_dir="$2"
    local script_dir="$3"
    
    local base_name=$(basename "$input_file" | sed 's/\.[^.]*$//')
    local output_file="${output_dir}/${base_name}_peacoqc.fcs"
    
    Rscript "${script_dir}/r_algorithms/run_peacoqc.R" "$input_file" "$output_file"
}

export -f run_peacoqc

# Run in parallel
# parallel --bar --jobs $MAX_PARALLEL run_peacoqc {} "$OUTPUT_DIR" "$SCRIPT_DIR" ::: "${FILES[@]}"

# Run in series
for file in "${FILES[@]}"; do
    run_peacoqc "$file" "$OUTPUT_DIR" "$SCRIPT_DIR"
done

echo "PeacoQC processing complete!"