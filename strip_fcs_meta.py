import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

import readfcs
import fcswrite
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def strip_fcs_metadata(input_fcs_path: Path, output_fcs_path: Path) -> None:
    """
    Reads an FCS file, strips most metadata, and writes a new FCS file.

    Keeps only essential parameter definitions ($PnN, $PnS) and relies on
    fcswrite to handle necessary structural keywords.

    Args:
        input_fcs_path: Path to the input FCS file.
        output_fcs_path: Path where the cleaned FCS file will be saved.
    """
    logging.info(f"Processing {input_fcs_path.name}...")

    # Read FCS file using readfcs
    adata = readfcs.read(str(input_fcs_path))

    # Extract metadata and data
    meta: Dict[str, Any] = adata.uns.get("meta", {})
    data = adata.to_df()

    num_params: int = len(adata.var_names)
    if num_params == 0:
        logging.error(f"Cannot process file with zero data columns: {input_fcs_path.name}")
        return

    # --- Extract channel names from adata.var ---
    chn_names: List[str] = []
    text_kw_pr: Dict[str, str] = {}

    for i, var_name in enumerate(adata.var_names):
        # Get channel name from adata.var
        raw_channel_name = str(var_name)

        # Process channel name: Remove FJ prefix if present.
        final_channel_name = raw_channel_name
        if final_channel_name.lower().startswith('fj'):
             final_channel_name = final_channel_name[2:]  # Remove first two chars FJ/fj

        chn_names.append(final_channel_name)

    # --- End of extraction ---

    logging.debug(f"Writing {output_fcs_path.name} with text_kw_pr: {text_kw_pr}")
    fcswrite.write_fcs(
        filename=str(output_fcs_path),
        chn_names=chn_names,
        data=data.values,
        text_kw_pr=text_kw_pr,
        compat_chn_names=True,
        compat_copy=True,
        compat_negative=True,
        compat_percent=True
    )
    logging.info(f"Successfully wrote cleaned file: {output_fcs_path.name}")


def main() -> None:
    """
    Main function to parse arguments and process FCS files in a directory.
    """
    parser = argparse.ArgumentParser(
        description="Strips metadata from FCS files in a directory, keeping only "
                    "channel names ($PnN) and stain names ($PnS)."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the input FCS files."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Directory where the cleaned FCS files will be saved."
    )

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        logging.error(f"Input directory not found: {input_dir}")
        return

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory set to: {output_dir}")

    fcs_files = list(input_dir.glob('*.fcs'))
    if not fcs_files:
        logging.warning(f"No .fcs files found in {input_dir}")
        return

    logging.info(f"Found {len(fcs_files)} FCS files to process.")

    for fcs_file_path in fcs_files:
        output_file_path = output_dir / fcs_file_path.name
        strip_fcs_metadata(fcs_file_path, output_file_path)

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()
