# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import os
import logging

from src.py.common.common import load_pc_dat


@click.command()
@click.option('--root-dir', type=str, help='Path to the preprocessed DSAI sequence to update', required=True)
@click.option('--parquet-file', type=str, help='Path to the parquet file to import NV labels from', required=True)
# @click.option('--output-dir',
#               type=str,
#               help='Path to the output target directory - if missing, will update the original sequence in place',
#               required=False,
#               default=None)
def dsai_import_labels(root_dir : str, output_dir : str, parquet_file : str):
    ''' Imports the label-related properties of a preprocessed DSAI sequence, replacing existing labels

    Important: - Importer currently only supports importing cuboid labels (following NV specs)
                 from parquet files, and can therefore only be used along with NV data.
    ''' 

    # Initialize the logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.info(f"Importing from {parquet_file} into {root_dir}")

if __name__ == "__main__":
    dsai_import_labels()
