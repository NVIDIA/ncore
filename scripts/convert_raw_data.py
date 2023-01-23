# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import logging

from src.dsai_internal.common.common import Config


@click.group()
@click.option('--root-dir', type=str, help="Path to the raw data sequences", required=True)
@click.option('--output-dir', type=str, help="Path where the converted data will be saved", required=True)
@click.option("--debug", is_flag=True, default=False, help="Enables debug logging outputs")
@click.pass_context
def cli(ctx, *_, **kwargs):
    """Data Preprocessing Pipeline
    
    Source data format is selected via subcommands, for which dedicated options can be specified.
    
    Example invocation for 'NV maglev' data
    
    \b
    ./convert_raw_data.py 
      --root-dir <FOLDER WITH SOURCE DATASETS>
      --output-dir <FOLDER DATA WILL BE PRODUCED>
      nvidia-maglev-v3
    """
    # Create a config object and remember it as the context object. From
    # this point onwards other commands can refer to it by using the
    # @click.pass_context decorator.
    ctx.obj = Config(kwargs)

    # Initialize basic top-level logger configuration
    logging.basicConfig(level=logging.DEBUG if ctx.obj.debug else logging.INFO,
                        format='<%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d|%(name)s> %(message)s')


@cli.command()
@click.option('--start-timestamp-us', type=int, default=None, help="If provided, the start timestamp to restrict processing to")
@click.option('--end-timestamp-us', type=int, default=None, help="If provided, the end timestamp to restrict processing to")
@click.pass_context
def nvidia_deepmap_v3(ctx, *_, **kwargs):
    """NVIDIA-specific data conversion (V3 format, based on DeepMap tracks)"""

    from src.dsai_internal.data_converter.nvidia_deepmap3 import NvidiaDeepmapConverter

    config = ctx.obj  # Extend base config with command-specific options
    config += kwargs

    NvidiaDeepmapConverter.convert(config)


@cli.command()
@click.option('--seek-sec', type=click.FloatRange(min=0.0, max_open=True), help="Time to skip for the dataset conversion (in seconds)")
@click.option('--duration-sec', type=click.FloatRange(min=0.0, max_open=True), help="Restrict total duration of the dataset conversion (in seconds)")
@click.option('--shard-id', type=click.IntRange(min=0, max_open=True), default=0, help="Shard id in [0,N-1] controlling uniform dataset subset processing")
@click.option('--shard-count', type=click.IntRange(min=1, max_open=True), default=1, help="Total number of shards N to performing full dataset processing")
@click.option('--egomotion-file', type=str, help="If provided, overwrite default egomotion file location", default=None)
@click.option('--skip-dynamic-flag', is_flag=True, default=False, help="Skip lidar dynamic flag computation to improve performance")
@click.pass_context
def nvidia_maglev_v3(ctx, *_, **kwargs):
    """NVIDIA-specific data conversion (V3 format, based on Maglev data extraction)"""

    from src.dsai_internal.data_converter.nvidia_maglev3 import NvidiaMaglevConverter

    config = ctx.obj  # Extend base config with command-specific options
    config += kwargs

    NvidiaMaglevConverter.convert(config)


if __name__ == '__main__':
    cli(show_default=True)
