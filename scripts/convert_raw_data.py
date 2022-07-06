#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click

from src.dataset_converter import DataConverter
from src.dataset_converter.waymo_open import WaymoConverter
from src.dataset_converter.nvidia_deepmap import NvidiaDeepMapConverter
from src.dataset_converter.nvidia_maglev import NvidiaMaglevConverter

@click.group()
@click.option('--root-dir', type=str, help="Path to the raw data sequences", required=True)
@click.option('--output-dir', type=str, help="Path where the converted data will be saved", required=True)
@click.option('--n-proc', default=1, type=int, help='Number of processes to spawn')
@click.option('--semantic-seg', is_flag=True, default=False, help="Infer the semantic segmention for all camera images")
@click.option('--instance-seg', is_flag=True, default=False, help="Infer the instance segmention for all camera images")
@click.option('--surface-rec', is_flag=True, default=False, help="Reconstruct the static background mesh")
@click.version_option('0.1')
@click.pass_context
def cli(ctx, *_, **kwargs):
    """Data Preprocessing Pipeline
    
    Source data format is selected via subcommands, for which dedicated options can be specified.
    
    Example invocation for 'waymo' data
    
    \b
    ./convert_raw_data.py 
      --root-dir <FOLDER WITH SOURCE DATASETS>
      --output-dir <FOLDER DATA WILL BE PRODUCED>
      waymo
      --ref-projections
    """
    # Create a DataConverter config object and remember it as the context object. From
    # this point onwards other commands can refer to it by using the
    # @click.pass_context decorator.
    ctx.obj = DataConverter.Config(kwargs)


@cli.command()
@click.option('--ref-projections', is_flag=True, default=False, help="Store reference point-cloud to image projections (explicitly available in waymo data)")
@click.pass_context
def waymo(ctx, *_, **kwargs):
    """Waymo-specific data conversion"""
    config = ctx.obj  # Extend base config with command-specific options
    config += kwargs
    WaymoConverter(config).convert()


@cli.command()
# TODO(#9): add --seek-sec / --duration-sec options similar to nvidia_maglev subcommand
@click.pass_context
def nvidia_deepmap(ctx, *_, **kwargs):
    """NVIDIA-specific data conversion (based on DeepMap tracks)"""
    config = ctx.obj  # Extend base config with command-specific options
    config += kwargs
    NvidiaDeepMapConverter(config).convert()


@cli.command()
@click.option('--seek-sec', type=click.FloatRange(min=0.0, max_open=True), help="Time to skip for the dataset conversion (in seconds)")
@click.option('--duration-sec', type=click.FloatRange(min=0.0, max_open=True), help="Restrict total duration of the dataset conversion (in seconds)")
@click.pass_context
def nvidia_maglev(ctx, *_, **kwargs):
    """NVIDIA-specific data conversion (based on Maglev data extraction)"""
    config = ctx.obj  # Extend base config with command-specific options
    config += kwargs
    NvidiaMaglevConverter(config).convert()


if __name__ == '__main__':
    cli(show_default=True)
