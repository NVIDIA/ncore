# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import logging

from ncore.impl.common.common import Config

logger = logging.getLogger(__name__)

@click.command()
@click.option('--root-dir', type=str, help="Path to the raw data sequences", required=True)
@click.option('--output-dir', type=str, help="Path where the converted data will be saved", required=True)
@click.option('--seek-sec', type=click.FloatRange(min=0.0, max_open=True), help="Time to skip for the dataset conversion (in seconds)")
@click.option('--duration-sec', type=click.FloatRange(min=0.0, max_open=True), help="Restrict total duration of the dataset conversion (in seconds)")
@click.option('--shard-id', type=click.IntRange(min=0, max_open=True), default=0, help="Shard id in [0,N-1] controlling uniform dataset subset processing")
@click.option('--shard-count', type=click.IntRange(min=1, max_open=True), default=1, help="Total number of shards N to performing full dataset processing")
@click.option("--wds-shard-maxsize-mib", type=click.IntRange(min=1, max_open=True), default=4096, help="Maximum size per webdataset shard in MiB")
@click.option("--wds-shard-maxsamples", type=click.IntRange(min=1, max_open=True), default=100000, help="Target number of samples per webdataset shard")
@click.option("--min-speed-km-h", type=click.FloatRange(min=0.0, max_open=True), default=10, help="If provided, the minimum required speed to filter out low-motion data (kilometer/hour)")
@click.pass_context
def sd_shards(ctx, *_, **kwargs):
    """NVIDIA-specific data conversion (Stable-Diffusion data, based on Maglev data extraction)"""

    from nvidia_maglev_sd import NvidiaMaglevConverter

    config = Config(kwargs)

    NvidiaMaglevConverter.convert(config)

if __name__ == '__main__':
    sd_shards(show_default=True)
