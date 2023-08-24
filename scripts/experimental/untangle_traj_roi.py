# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging
import glob
import re
import json

from pathlib import Path
from dataclasses import dataclass

import click
import tqdm
import numpy as np
import polyscope as ps

from ncore.impl.common.nvidia_utils import load_maglev_egomotion
from ncore.impl.common.transformations import se3_inverse


@dataclass(kw_only=True, slots=True, frozen=True)
class CLIParams:
    ''' Parameters passed to CLI '''
    source_dir: str
    roi_id: str
    skip_sessions: tuple[str]
    untangle_max_split_time_sec: float
    max_chunk_time_sec: float | None
    output_dir: str | None
    visualize: bool
    verbose: bool


@dataclass(kw_only=True, slots=True, frozen=True)
class TrajectoryChunk:
    source_session_id: str
    source_egomotion_json: dict
    source_start_pose_idx: int
    source_stop_pose_idx: int
    chunk_in_session_id: int
    common_poses: np.ndarray
    timestamps_us: np.ndarray

    def __post_init__(self):
        assert len(self.common_poses) == len(self.timestamps_us)
        assert len(self.common_poses) > 1

    def length_sec(self) -> float:
        return (self.timestamps_us[-1] - self.timestamps_us[0]) / 1e6


@click.command()
@click.option('--source-dir',
              type=str,
              help='Path to the source folder with .json trajectories to untangle',
              required=True)
@click.option('--roi-id', type=str, help='Common ROI identifier', required=True)
@click.option(
    '--untangle-max-split-time-sec',
    type=float,
    default=0.2,
    help=
    'Max consecutive time difference between poses to split / untangle trajectories into individual chunks [due to DeepMap\'s data mixing these into a single representation] (sec)',
    required=True)
@click.option('--max-chunk-time-sec', type=float, default=None, help='Max individual chunk time (sec)')
@click.option('--skip-session',
              'skip_sessions',
              multiple=True,
              type=str,
              help='Session-id\'s to be skipped',
              default=None)
@click.option('--output-dir', type=str, help='If provided, the path to the output folder to export untangled chunks to')
@click.option("--visualize", is_flag=True, default=False, help="Enable rendering of untangled chunks")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logging outputs")
def untangle_traj_roi(**kwargs) -> None:
    """A tool to untangle, filter, re-export and visualize deepmap ROI-associated egomotion data"""

    # Parse params
    params = CLIParams(**kwargs)

    # Initialize the logger
    logging.basicConfig(level=logging.DEBUG if params.verbose else logging.INFO)

    # Use identity sensor extrinsics for simplicity (poses are only used for rendering, extend as necessary)
    T_rig_sensors: dict[str, np.ndarray] = {
        'lidar:gt:top:p128:v4p5': np.identity(4, dtype=np.float32),
        'lidar:gt:top:p128': np.identity(4, dtype=np.float32)
    }

    #  load all input deepmap egomotion files and untangle into trajectories
    T_rig_world_base: None | np.ndarray = None
    trajectory_chunks: list[TrajectoryChunk] = []
    for json_path in tqdm.tqdm([Path(path) for path in glob.glob(params.source_dir + '/*.json')],
                               desc=f"Parsing tangled input trajectories"):
        logging.debug(json_path)

        # load session ID
        if match := re.search(r'(\w{8}-\w{4}-\w{4}-\w{4}-\w{12}).*\.json', str(json_path)):
            session_id = match[1]
        else:
            raise ValueError("Unable to determine trustable session_id")

        logging.debug(session_id)

        if session_id in params.skip_sessions:
            logging.info(f"Skipping session {session_id} from skip-list")
            continue

        # load egomotion
        with open(json_path, "r") as fp:
            egomotion_json = json.load(fp)

        global_T_rig_worlds, T_rig_world_timestamps_us = load_maglev_egomotion(
            T_rig_sensors,  # type: ignore
            json_path)  # type: ignore

        if T_rig_world_base is None:
            # use first rig_world base as anchor for all trajectories
            T_rig_world_base = global_T_rig_worlds[0]
        common_T_rig_worlds = se3_inverse(T_rig_world_base) @ global_T_rig_worlds

        timestamp_differences_us = np.diff(T_rig_world_timestamps_us)
        timestamp_differences_sec = timestamp_differences_us / 1e6
        logging.debug(timestamp_differences_sec.max())

        # untangle by splitting at long pauses in consecutive poses
        splitpoints = np.where(timestamp_differences_sec > params.untangle_max_split_time_sec
                               )[0] + 1  # increment by one to use "following" index as stop index
        if len(splitpoints):
            start_pose = 0
            for chunk_in_session_id, splitpoint in enumerate(splitpoints):
                # make sure there are at least two pose in a chunk
                if splitpoint - start_pose > 1:
                    trajectory_chunks.append(
                        TrajectoryChunk(source_session_id=session_id,
                                        source_egomotion_json=egomotion_json,
                                        source_start_pose_idx=start_pose,
                                        source_stop_pose_idx=splitpoint,
                                        chunk_in_session_id=chunk_in_session_id,
                                        common_poses=common_T_rig_worlds[start_pose:splitpoint],
                                        timestamps_us=np.array(T_rig_world_timestamps_us[start_pose:splitpoint])))
                start_pose = splitpoint
        else:
            # full session
            trajectory_chunks.append(
                TrajectoryChunk(source_session_id=session_id,
                                source_egomotion_json=egomotion_json,
                                source_start_pose_idx=0,
                                source_stop_pose_idx=len(global_T_rig_worlds),
                                chunk_in_session_id=0,
                                common_poses=common_T_rig_worlds,
                                timestamps_us=np.array(T_rig_world_timestamps_us)))

    # filter based on max-chunk-time
    if params.max_chunk_time_sec:
        for i, trajectory_chunk in enumerate(list(trajectory_chunks)):
            if (chunk_time_sec := trajectory_chunk.length_sec()) > params.max_chunk_time_sec:
                logging.info(f'filtering out chunk {i} with chunk time {chunk_time_sec}sec')
                trajectory_chunks.remove(trajectory_chunk)

    if params.visualize:
        ps.init()
        ps.set_up_dir("z_up")

        for i, trajectory_chunk in enumerate(trajectory_chunks):
            world_common_positions = trajectory_chunk.common_poses[:, :3, 3]

            curve = ps.register_curve_network(
                '_'.join([params.roi_id, trajectory_chunk.source_session_id,
                          str(trajectory_chunk.chunk_in_session_id)]),
                world_common_positions,
                "line",
                enabled=True,
                radius=0.001,
            )
            curve.add_scalar_quantity("normalized_time",
                                      np.linspace(0.0, 1.0, num=len(trajectory_chunk.timestamps_us)),
                                      enabled=True,
                                      cmap='jet')

        ps.show()

    if params.output_dir is None:
        # exit early if not exporting
        return

    # store results
    output_path = Path(params.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for trajectory_chunk in trajectory_chunks:
        # adapt source egomotion to only contain subset of poses
        egomotion_json = trajectory_chunk.source_egomotion_json.copy()

        if 'poses' in egomotion_json:
            poses_key = 'poses'  # Deepmap egomotin json format
        else:
            poses_key = 'tf_frame_world'  # new egomotion json format

        egomotion_json[poses_key] = egomotion_json[poses_key][trajectory_chunk.source_start_pose_idx:trajectory_chunk.
                                                              source_stop_pose_idx]

        output_chunk_path = output_path / (
            '_'.join([params.roi_id, trajectory_chunk.source_session_id,
                      str(trajectory_chunk.chunk_in_session_id)]) + '.json')
        with open(output_chunk_path, 'w') as fp:
            fp.write(json.dumps(egomotion_json))

        logging.info(f'wrote {output_chunk_path}')


if __name__ == '__main__':
    untangle_traj_roi(show_default=True)
