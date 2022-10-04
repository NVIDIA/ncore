#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from asyncio.proactor_events import _ProactorBasePipeTransport
import click
import debugpy
import logging
import os
import glob

from src.py.dataset_converter import DataConverter

class SegmentationConverter(DataConverter):  
    """
    DISCLAIMER: THIS SOURCE CODE IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY.
    IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT ZAN GOJCIC zgojcic@nvidia.com. 
    """  

    def __init__(self, config):
        
        config.output_dir = config.root_dir
        config.root_dir = None
        super().__init__(config)

        self.sequence_pathnames = sorted(glob.glob(os.path.join(self.output_dir, '*')))

    def convert_one(self, sequence_path):
        sequence_name = sequence_path.split('/')[-1]
        return [sequence_name] 

logger = logging.getLogger(__name__)

@click.command()
@click.option('--root-dir', type=str, help="Path to the raw data sequences", required=True)
@click.option('--semantic-seg', is_flag=True, default=False, help="Infer the semantic segmention for all camera images")
@click.option('--instance-seg', is_flag=True, default=False, help="Infer the instance segmention for all camera images")
@click.option("--debug", is_flag=True, default=False, help="Enables a debugpy client to connect to the port specified by --debug-port")
@click.option("--debug-wait-for-client", is_flag=True, default=False, help="Enables a debugpy client to connect to the port specified by --debug-port and waits for a client to connect on start-up")
@click.option("--debug-port", default=5678, type=int, help="Configure the TCP port to use for debugging")
@click.version_option('0.1')
@click.pass_context
def extract_segmentation(ctx, *_, **kwargs):
    """Semantic/Instance Segmentation Extraction for Processed Data

    Example invocation
    
    \b
    bazel run //scripts:dsai_extract_segmentation -- 
    --root-dir <FOLDER WITH PROCESSED DATASETS> 
    --semantic-seg
    --instance-seg
    """
    # Create a DataConverter config object and remember it as the context object. From
    # this point onwards other commands can refer to it by using the
    # @click.pass_context decorator.
    ctx.obj = DataConverter.Config(kwargs)

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

    extractor = SegmentationConverter(ctx.obj)
    
    for sequence_pathname in extractor.sequence_pathnames:
        sequence_name = sequence_pathname.split('/')[-1]
        
        if ctx.obj.semantic_seg:
            extractor.run_semantic_segmentation(sequence_name)
        if ctx.obj.instance_seg:
            extractor.run_instance_segmentation(sequence_name)

if __name__ == '__main__':
    extract_segmentation(show_default=True)


