#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import debugpy
import logging

from src.py.common.common import Config

logger = logging.getLogger(__name__)

@click.group()
@click.option('--root-dir', type=str, help="Path to the raw data sequences", required=True)
@click.option('--output-dir', type=str, help="Path where the converted data will be saved", required=True)
@click.option("--debug", is_flag=True, default=False, help="Enables a debugpy client to connect to the port specified by --debug-port")
@click.option("--debug-wait-for-client", is_flag=True, default=False, help="Enables a debugpy client to connect to the port specified by --debug-port and waits for a client to connect on start-up")
@click.option("--debug-port", default=5678, type=int, help="Configure the TCP port to use for debugging")
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
    # Create a config object and remember it as the context object. From
    # this point onwards other commands can refer to it by using the
    # @click.pass_context decorator.
    ctx.obj = Config(kwargs)

    # Conditionally enable debugging
    if ctx.obj.debug or ctx.obj.debug_wait_for_client:
        # Note: enabling debug impacts the performance of system calls and Python code execution.
        logger.info("Listening for incoming debug connection on port {}".format(
            ctx.obj.debug_port))
        debugpy.listen(("0.0.0.0", ctx.obj.debug_port))

        if ctx.obj.debug_wait_for_client:
            logger.info("Waiting for incoming debug connection on port {}".format(
                ctx.obj.debug_port))
            # Block until a client connects
            debugpy.wait_for_client()


@cli.command()
@click.option('--ref-projections', is_flag=True, default=False, help="Store reference point-cloud to image projections (explicitly available in waymo data)")
@click.pass_context
def waymo(ctx, *_, **kwargs):
    """Waymo-specific data conversion"""

    from src.py.data_converter.waymo_open import WaymoConverter

    config = ctx.obj  # Extend base config with command-specific options
    config += kwargs

    WaymoConverter(config).convert()


@cli.command()
@click.option('--start-timestamp-us', type=int, default=None, help="If provided, the start timestamp to restrict processing to")
@click.option('--end-timestamp-us', type=int, default=None, help="If provided, the end timestamp to restrict processing to")
@click.pass_context
def nvidia_deepmap(ctx, *_, **kwargs):
    """NVIDIA-specific data conversion (based on DeepMap tracks)"""

    from src.py.data_converter.nvidia_deepmap import NvidiaDeepMapConverter

    config = ctx.obj  # Extend base config with command-specific options
    config += kwargs

    NvidiaDeepMapConverter(config).convert()


@cli.command()
@click.option('--seek-sec', type=click.FloatRange(min=0.0, max_open=True), help="Time to skip for the dataset conversion (in seconds)")
@click.option('--duration-sec', type=click.FloatRange(min=0.0, max_open=True), help="Restrict total duration of the dataset conversion (in seconds)")
@click.option('--multiprocessing-camera', is_flag=True, default=False, help="Perform camera data conversion with multiprocessing if enabled")
@click.option('--multiprocessing-lidar', is_flag=True, default=False, help="Perform lidar data conversion with multiprocessing if enabled")
@click.option('--max-processes', default=None, type=int, help="If provided, the upper bound for processes to start")
@click.option('--shard-id', type=click.IntRange(min=0, max_open=True), default=0, help="Shard id in [0,N-1] controlling uniform dataset subset processing")
@click.option('--shard-count', type=click.IntRange(min=1, max_open=True), default=1, help="Total number of shards N to performing full dataset processing")
@click.option('--symlink-camera-frames', is_flag=True, default=False, help="Symlink camera frames instead of copying files if enabled")
@click.option('--compress-lidar', is_flag=True, default=False, help="Compress lidar frame data if enabled")
@click.option('--egomotion-file', type=str, help="If provided, overwrite default egomotion file location", default=None)
@click.option('--skip-dynamic-flag', is_flag=True, default=False, help="Skip lidar dynamic flag computation to improve performance")
@click.pass_context
def nvidia_maglev(ctx, *_, **kwargs):
    """NVIDIA-specific data conversion (V1 format, based on Maglev data extraction)"""
    
    from src.py.data_converter.nvidia_maglev import NvidiaMaglevConverter

    config = ctx.obj  # Extend base config with command-specific options
    config += kwargs

    NvidiaMaglevConverter(config).convert()

@cli.command()
@click.option('--seek-sec', type=click.FloatRange(min=0.0, max_open=True), help="Time to skip for the dataset conversion (in seconds)")
@click.option('--duration-sec', type=click.FloatRange(min=0.0, max_open=True), help="Restrict total duration of the dataset conversion (in seconds)")
@click.option('--multiprocessing-camera', is_flag=True, default=False, help="Perform camera data conversion with multiprocessing if enabled")
@click.option('--multiprocessing-lidar', is_flag=True, default=False, help="Perform lidar data conversion with multiprocessing if enabled")
@click.option('--max-processes', default=None, type=int, help="If provided, the upper bound for processes to start")
@click.option('--shard-id', type=click.IntRange(min=0, max_open=True), default=0, help="Shard id in [0,N-1] controlling uniform dataset subset processing")
@click.option('--shard-count', type=click.IntRange(min=1, max_open=True), default=1, help="Total number of shards N to performing full dataset processing")
@click.option('--symlink-camera-frames', is_flag=True, default=False, help="Symlink camera frames instead of copying files if enabled")
@click.option('--compress-lidar', is_flag=True, default=False, help="Compress lidar frame data if enabled")
@click.option('--egomotion-file', type=str, help="If provided, overwrite default egomotion file location", default=None)
@click.option('--skip-dynamic-flag', is_flag=True, default=False, help="Skip lidar dynamic flag computation to improve performance")
@click.pass_context
def nvidia_maglev_v2(ctx, *_, **kwargs):
    """NVIDIA-specific data conversion (V2 format, based on Maglev data extraction)"""
    
    from src.py.data_converter.v2.nvidia_maglev import NvidiaMaglevConverter

    config = ctx.obj  # Extend base config with command-specific options
    config += kwargs

    NvidiaMaglevConverter.convert(config)


if __name__ == '__main__':
    cli(show_default=True)
