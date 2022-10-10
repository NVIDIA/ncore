# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from socket import PF_RDS
import click
import logging
import os
import glob

from src.py.common.common import NV_CAMERAS, WAYMO_CAMERAS
from src.py.deps.instance_segmentation import run_instance_segmentation
from src.py.deps.semantic_segmentation import run_semantic_segmentation

logger = logging.getLogger(__name__)

@click.group()
@click.option('--root-dir', type=str, help="Path to the raw data sequences", required=True)
@click.option('--cameras', '-c', multiple=True, type=int, help='Cameras to be used (Multiple value option, -1 denotes all)', default=[-1])  
@click.option('--semantic-seg', is_flag=True, default=False, help="Perform semantic segmentation")
@click.option('--instance-seg', is_flag=True, default=False, help="Perform instance segmentation") 
@click.option('--start-frame', type=int, help="The starting frame of the sequence", default=0)
@click.option('--stop-frame', type=int, help="The end frame of the sequence", default=-1)
@click.option("--index-digits", type=int, help="The number of integer digits to pad counters in output filenames to", default=6)
@click.version_option('0.1')
@click.pass_context
def cli(ctx, *_, **kwargs):
    ctx.obj = kwargs

@cli.command()
@click.pass_context
def waymo(ctx, *_, **kwargs):

    assert os.path.exists(ctx.obj['root_dir']), "Provided root_dir path doesn't exist."
    assert (ctx.obj['start_frame'] < ctx.obj['stop_frame'] or ctx.obj['stop_frame'] == -1), "stop frame must be larger than start frame"

    # Check if the selected cameras are valid
    if len(ctx.obj['cameras']) == 1 and ctx.obj['cameras'][0] == -1: 
        ctx.obj['cameras'] = WAYMO_CAMERAS
    else:
        for cam in ctx.obj['cameras']:
            assert str(cam).zfill(2) in WAYMO_CAMERAS, "Invalid camera selected for Waymo dataset."


    for cam in ctx.obj['cameras']:
        imgs = sorted(glob.glob(os.path.join(ctx.obj['root_dir'], f'images/image_{str(cam).zfill(2)}', '*.jpeg')))

        start_frame = ctx.obj['start_frame']
        stop_frame = ctx.obj['stop_frame']
        if ctx.obj['start_frame']  > len(imgs):
            logging.warning(f"The start frame is larger than the number of images. All images will be processed")
            start_frame = 0
            stop_frame = len(imgs)

        elif ctx.obj['stop_frame'] != -1 and ctx.obj['stop_frame'] > len(imgs):
            logging.warning(f"The stop frame is larger than the number of images. All frames larger than start frame will be processed")
            stop_frame = len(imgs)
        
        elif ctx.obj['stop_frame'] == -1:
            stop_frame = len(imgs)
            
        if ctx.obj['semantic_seg']:
            run_semantic_segmentation(imgs[start_frame:stop_frame + 1], ctx.obj['index_digits'])

        if ctx.obj['instance_seg']:
            run_instance_segmentation(imgs[start_frame:stop_frame + 1])

@cli.command()
@click.pass_context
def nvidia(ctx, *_, **kwargs):

    assert os.path.exists(ctx.obj['root_dir']), "Provided root_dir path doesn't exist."
    assert (ctx.obj['start_frame'] < ctx.obj['stop_frame'] or ctx.obj['stop_frame'] == -1), "stop frame must be larger than start frame"

    # Check if the selected cameras are valid
    if len(ctx.obj['cameras']) == 1 and ctx.obj['cameras'][0] == -1: 
        ctx.obj['cameras'] = NV_CAMERAS
    else:
        for cam in ctx.obj['cameras']:
            assert str(cam).zfill(2) in NV_CAMERAS, "Invalid camera selected for Nvidia dataset."

    for cam in ctx.obj['cameras']:
        imgs = sorted(glob.glob(os.path.join(ctx.obj['root_dir'], f'images/image_{str(cam).zfill(2)}', '*.jpeg')))

        start_frame = ctx.obj['start_frame']
        stop_frame = ctx.obj['stop_frame']
        if ctx.obj['start_frame']  > len(imgs):
            logging.warning(f"The start frame is larger than the number of images. All images will be processed")
            start_frame = 0
            stop_frame = len(imgs)

        elif ctx.obj['stop_frame'] != -1 and ctx.obj['stop_frame'] > len(imgs):
            logging.warning(f"The stop frame is larger than the number of images. All frames larger than start frame will be processed")
            stop_frame = len(imgs)
         
        elif ctx.obj['stop_frame'] == -1:
            stop_frame = len(imgs)

        if ctx.obj['semantic_seg']:
            run_semantic_segmentation(imgs[start_frame:stop_frame + 1], ctx.obj['index_digits'])

        if ctx.obj['instance_seg']:
            run_instance_segmentation(imgs[start_frame:stop_frame + 1])

if __name__ == "__main__":
    cli(show_default=True)