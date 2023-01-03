# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import io
import json
import re
import lzma
import tarfile
import logging

from types import SimpleNamespace
from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache, cache
from typing import BinaryIO, Iterator, Literal, NamedTuple, Optional, Union
from threading import RLock

import numpy as np
import zarr
import numcodecs
import dataclasses_json
import PIL.Image as PILImage

import src.dsai_internal.common.common as common
import src.dsai_internal.common.transformations as transformations

from . import types, util

VERSION = '3.0.0'
CAMERAS_BASE_GROUP = 'cameras'
LIDARS_BASE_GROUP = 'lidars'
RADARS_BASE_GROUP = 'radars'


class IndexedTarStore(zarr._storage.store.Store):
    """ A zarr store over *indexed* tar files
    
    Parameters
    ----------
    tar_path : string
        Location of the tar file (needs to end with '.tar').
        A corresponding index table file will be stored at the same path with a '.taridx.xz' suffix.
    mode : string, optional
        One of 'r' to read an existing file, or 'w' to truncate and write a new
        file.

    After modifying a IndexedTarStore, the ``close()`` method must be called, otherwise
    essential data will not be written to the underlying files. The IndexedTarStore
    class also supports the context manager protocol, which ensures the ``close()``
    method is called on leaving the context, e.g.::

        >>> with IndexedTarStore('data/array.tar', mode='w') as store:
        ...     z = zarr.zeros((10, 10), chunks=(5, 5), store=store)
        ...     z[...] = 42
        ...     # no need to call store.close()
    
    """

    _erasable = False

    @dataclass
    class TarRecord(dataclasses_json.DataClassJsonMixin):
        """ A file record within a tar file """

        offset_data: int
        size: int

    @dataclass
    class TarRecordIndex(dataclasses_json.DataClassJsonMixin):
        """ All file records within a tar file """

        records: dict[str, IndexedTarStore.TarRecord] = field(default_factory=dict)

    def __init__(self, tar_path: Union[str, Path], mode: Literal['r', 'w'] = 'r'):

        assert mode in ['r', 'w']

        # store properties
        self.tar_path = Path(tar_path).absolute()
        assert self.tar_path.suffix == '.tar', f"{tar_path} is not a '.tar' file path"

        self.index_path = self.tar_path.with_suffix('.taridx.xz')
        
        self.mode = mode

        # Current understanding is that tarfile module in stdlib is not thread-safe,
        # and so locking is required for both read and write. However, this has not
        # been investigated in detail, perhaps no lock is needed if mode='r'.
        self.mutex = RLock()

        # open tar file and file object
        if self.mode == 'w':
            # require file to be both writeable and readable
            self.tar_file = tarfile.TarFile(fileobj=open(self.tar_path, 'wb+'), mode=self.mode)
        else:
            self.tar_file = tarfile.TarFile(fileobj=open(self.tar_path, 'rb'), mode=self.mode)
        self.tar_file_object: BinaryIO = self.tar_file.fileobj # type: ignore

        # init / load index table
        if mode == 'r':
            # load table (SOA)
            with lzma.open(self.index_path, 'rt') as f:
                table = json.load(f)
                items = table['items']
                offset_datas = table['offset_datas']
                sizes = table['sizes']

            # create record map
            self.file_index = self.TarRecordIndex(
                {item: self.TarRecord(offset_datas[i], sizes[i])
                 for i, item in enumerate(items)})

            del (table, items, offset_datas, sizes)
        else:
            self.file_index = self.TarRecordIndex()

    def __delitem__(self, _: str):
        raise NotImplementedError('Deleting items is not supported')

    def __iter__(self) -> Iterator[str]:
        with self.mutex:
            return iter(self.file_index.records.keys())

    def __len__(self) -> int:
        with self.mutex:
            return len(self.file_index.records)

    def __getitem__(self, item: str) -> bytes:

        record = self.file_index.records[item]  # raises KeyError if not in archive

        # Remember current tar file position
        current_position = self.tar_file_object.tell()

        # Read the value
        self.tar_file_object.seek(record.offset_data)
        value = self.tar_file_object.read(record.size)

        # Return tar file to previous location
        self.tar_file_object.seek(current_position)

        return value

    def __setitem__(self, item: str, value):

        if self.mode != 'w':
            raise zarr.errors.ReadOnlyError

        with self.mutex:
            if item in self.file_index.records:
                raise ValueError(f'{item} already exists, update is not supported')

            value_bytes: bytes = numcodecs.compat.ensure_bytes(value)

            record = self.TarRecord(
                # Start of data in tar file (current tar file position + header-size)
                self.tar_file_object.tell() + tarfile.BLOCKSIZE,
                # Length of the data
                len(value_bytes))

            tarinfo = tarfile.TarInfo(item)
            tarinfo.size = record.size

            self.tar_file.addfile(tarinfo, fileobj=io.BytesIO(value_bytes))

            self.file_index.records[item] = record

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """ Needs to be called after finishing updating the store """
        with self.mutex:
            self.tar_file.close()
            self.tar_file_object.close()

            if self.mode == 'w':
                # Store index table as SOA (sorted by offset)
                table = [(item, record.offset_data, record.size) for (item, record) in self.file_index.records.items()]
                items, offset_datas, sizes = list(zip(*sorted(table, key=lambda data: data[1])))

                with lzma.open(self.index_path, 'wt') as f:
                    json.dump({'items': items, 'offset_datas': offset_datas, 'sizes': sizes}, f)

class ContainerDataWriter:
    ''' DataWriter implementing format-specific serialization specifications to store V3 / zarr data generated by data-converters '''
    def __init__(self, output_dir_path: Path,
                 container_name: str,
                 camera_ids: list[str],
                 lidar_ids: list[str],
                 radar_ids: list[str],
                 calibration_type: str,
                 egomotion_type: str,
                 sequence_id: str,
                 shard_id: int,
                 shard_count: int):
        ''' 
        Instantiate data writer and initialize the default data groups for a given sequence and sensor IDs
        '''

        self.camera_ids = camera_ids
        self.lidar_ids = lidar_ids
        self.radar_ids = radar_ids

        # Check that sensor-id's are unique across sensor types
        assert len(set(self.camera_ids + self.lidar_ids + self.radar_ids)) == len(self.camera_ids) + len(
            self.lidar_ids) + len(self.radar_ids), 'sensor id\'s are not unique'

        self.output_dir_path = output_dir_path
        self.container_name = container_name

        self.sequence_id = sequence_id
        self.shard_id = shard_id
        self.shard_count = shard_count

        # Initialize container file (indexed tar file)
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
        self.container_store = IndexedTarStore(self.output_dir_path / f'{self.container_name}.zarr.tar', mode='w')
        self.container_root = zarr.group(store=self.container_store)

        # Store dataset associated meta-data
        self.container_root.attrs.put({
            'version': VERSION,
            'egomotion_type': egomotion_type,
            'calibration_type': calibration_type,
            'camera_ids': self.camera_ids,
            'lidar_ids': self.lidar_ids,
            'radar_ids': self.radar_ids,
            'sequence_id': self.sequence_id,
            'shard_id': self.shard_id,
            'shard_count': self.shard_count
        })

        # Create sensor groups
        cameras_grp = self.container_root.require_group(CAMERAS_BASE_GROUP)
        for camera_id in self.camera_ids:
            cameras_grp.require_group(camera_id)

        lidars_grp = self.container_root.require_group(LIDARS_BASE_GROUP)
        for lidar_id in self.lidar_ids:
            lidars_grp.require_group(lidar_id)

        radars_grp = self.container_root.require_group(RADARS_BASE_GROUP)
        for radar_id in self.radar_ids:
            radars_grp.require_group(radar_id)

        # Store initial non-successful shard state
        self._store_shard_meta(False)

    def _store_shard_meta(self, successful: bool) -> None:
        # Store current shard success state (to distinguish successful / failed conversions without having to load zarr file)
        with open(
                self.output_dir_path /
                f'shard-meta-{self.sequence_id}-{util.padded_index_string(self.shard_id, index_digits=4)}.json',
                'w') as outfile:
            json.dump({'shard-id': self.shard_id, 'shard-count': self.shard_count, 'successful': successful}, outfile)

    # To be called after all data was added
    def finalize(self) -> None:

        # Make sure the shard file is consolidated
        zarr.consolidate_metadata(self.container_store)

        # Finish writing all files
        self.container_store.close()

        # Mark shard as successful
        self._store_shard_meta(True)

    # Individual 'store*' methods performing data sanity checks and serialize consistent output formats
    def store_poses(self, poses: types.Poses) -> None:
        poses_grp = self.container_root.require_group('poses')
        poses_grp.create_dataset('T_rig_world_base', data=poses.T_rig_world_base)
        poses_grp.create_dataset('T_rig_worlds', data=poses.T_rig_worlds)
        poses_grp.create_dataset('T_rig_world_timestamps_us', data=poses.T_rig_world_timestamps_us)

    def store_labels(self, track_labels: dict[str, types.TrackLabel]) -> None:
        output = {'track_labels': {k: v.to_dict() for (k, v) in track_labels.items()}}

        # TODO: add sanity checks on the final label structure before output

        self.container_root.require_group('labels').create_dataset('json', data=json.dumps(output))

    def store_camera_meta(
            self,
            camera_id: str,

            # all frame timestamps
            frame_timestamps_us: np.ndarray,

            # extrinsics
            T_sensor_rig: np.ndarray,

            # intrinsics
            camera_model_parameters: Union[types.FThetaCameraModelParameters, types.PinholeCameraModelParameters],

            # sensor constants
            mask_image: Optional[PILImage.Image]) -> None:
        assert T_sensor_rig.shape == (4, 4)
        assert T_sensor_rig.dtype == np.dtype('float32')
        assert frame_timestamps_us.ndim == 1
        assert frame_timestamps_us.dtype == np.dtype('uint64')

        # Store meta data
        camera_grp = self.container_root[CAMERAS_BASE_GROUP][camera_id]
        camera_grp.attrs.put({
            'T_sensor_rig': T_sensor_rig.tolist(),
            'camera_model_type': camera_model_parameters.type(),
            'camera_model_parameters': camera_model_parameters.to_dict()
        })

        # Store timestamps
        camera_grp.create_dataset('frame_timestamps_us', data=frame_timestamps_us)

        # Store mask if available
        if mask_image:
            with io.BytesIO() as buffer:
                FORMAT = 'png'
                mask_image.save(buffer, format=FORMAT, optimize=True)  # encodes as png
                self.container_root[CAMERAS_BASE_GROUP][camera_id].create_dataset(
                    'mask', data=np.asarray(buffer.getvalue()), compressor=None).attrs['format'] = FORMAT

    def store_camera_frame(
            self,

            # data indexing
            camera_id: str,
            continous_frame_index: int,

            # image data
            image_file_binary_data: bytes,
            image_file_format: str,

            # poses
            T_rig_worlds: np.ndarray,
            timestamps_us: np.ndarray) -> None:
        # sanity / consistency checks
        assert T_rig_worlds.shape == (2, 4, 4)
        assert T_rig_worlds.dtype == np.dtype('float32')
        assert timestamps_us.shape == (2, )
        assert timestamps_us.dtype == np.dtype('uint64')

        # Initialize frame
        continous_frame_index_string = util.padded_index_string(continous_frame_index)
        frame_group = self.container_root[CAMERAS_BASE_GROUP][camera_id].require_group(continous_frame_index_string)

        # Store image data and meta data
        frame_group.create_dataset('image', data=np.asarray(image_file_binary_data),
                                   compressor=None).attrs['format'] = image_file_format
        frame_group.attrs.put({
            'T_rig_worlds': T_rig_worlds.tolist(),
            'timestamps_us': timestamps_us.tolist(),
        })

    def store_lidar_meta(
            self,
            lidar_id: str,

            # all frame timestamps
            frame_timestamps_us: np.ndarray,

            # extrinsics
            T_sensor_rig: np.ndarray) -> None:
        assert T_sensor_rig.shape == (4, 4)
        assert T_sensor_rig.dtype == np.dtype('float32')
        assert frame_timestamps_us.shape[1:] == ()
        assert frame_timestamps_us.dtype == np.dtype('uint64')

        # Store meta data
        lidar_grp = self.container_root[LIDARS_BASE_GROUP][lidar_id]
        lidar_grp.attrs.put({'T_sensor_rig': T_sensor_rig.tolist()})

        # Store timestamps
        lidar_grp.create_dataset('frame_timestamps_us', data=frame_timestamps_us)

    def store_lidar_frame(
            self,

            # data indexing
            lidar_id: str,
            continous_frame_index: int,

            # point-cloud data
            xyz_s: np.ndarray,
            xyz_e: np.ndarray,
            intensity: np.ndarray,
            timestamp_us: np.ndarray,
            dynamic_flag: np.ndarray,

            # label data
            frame_labels: list[types.FrameLabel3],

            # poses
            T_rig_worlds: np.ndarray,
            timestamps_us: np.ndarray) -> None:
        # sanity / consistency checks
        assert xyz_s.shape[1] == 3
        assert xyz_s.dtype == np.dtype('float32')
        assert xyz_e.shape[1] == 3
        assert xyz_e.dtype == np.dtype('float32')
        assert intensity.ndim == 1
        assert intensity.dtype == np.dtype('float32')
        assert timestamp_us.ndim == 1
        assert timestamp_us.dtype == np.dtype('uint64')
        assert dynamic_flag.ndim == 1
        assert dynamic_flag.dtype == np.dtype('int8')
        num_points = xyz_s.shape[0]
        assert all((xyz_s.shape[0] == num_points, xyz_e.shape[0] == num_points, intensity.shape[0] == num_points,
                    timestamp_us.shape[0] == num_points, dynamic_flag.shape[0] == num_points))

        assert T_rig_worlds.shape == (2, 4, 4)
        assert T_rig_worlds.dtype == np.dtype('float32')
        assert timestamps_us.shape == (2, )
        assert timestamps_us.dtype == np.dtype('uint64')

        # Initialize frame
        continous_frame_index_string = util.padded_index_string(continous_frame_index)
        frame_group = self.container_root[LIDARS_BASE_GROUP][lidar_id].require_group(continous_frame_index_string)

        # Store frame data
        frame_group.create_dataset('xyz_s', data=xyz_s)
        frame_group.create_dataset('xyz_e', data=xyz_e)
        frame_group.create_dataset('intensity', data=intensity)
        frame_group.create_dataset('timestamp_us', data=timestamp_us)
        frame_group.create_dataset('dynamic_flag', data=dynamic_flag)
        frame_group.create_dataset('frame_labels',
                                   dtype=object,
                                   data=[frame_label.to_dict() for frame_label in frame_labels],
                                   object_codec=numcodecs.JSON())

        # Output frame meta data
        frame_group.attrs.put({
            'T_rig_worlds': T_rig_worlds.tolist(),
            'timestamps_us': timestamps_us.tolist(),
        })


class Sensor:
    ''' Provides access to generic data available to all sensor types '''
    class ShardFrame(NamedTuple):
        ''' References a specific frame in a specific shard '''
        shard_index: int
        shard_frame_index: int

    @lru_cache
    def _get_shard_frame(self, continous_frame_index: int) -> ShardFrame:
        ''' For a given continous-frame, determine the corresponding shard-frame '''
        assert continous_frame_index >= 0 and continous_frame_index < self._shard_frame_map[-1], IndexError

        shard_index = int(np.searchsorted(self._shard_frame_map[1:], continous_frame_index, side='right'))
        shard_frame = int(continous_frame_index - self._shard_frame_map[shard_index])

        return self.ShardFrame(shard_index, shard_frame)

    @lru_cache
    def _get_continous_frame(self, shard_frame: ShardFrame) -> int:
        ''' For a given shard-frame, determine the corresponding continous-frame '''
        return self._shard_frame_map[shard_frame.shard_index] + shard_frame.shard_frame_index

    def __init__(self, sensor_id: str, sensor_group: str, shard_roots: list[zarr.Group]):
        assert len(shard_roots), "Require at least a single shard"

        self._sensor_id = sensor_id
        self._sensor_group = sensor_group

        self._shard_roots = shard_roots

        # Load all shard meta-data
        shard_sensor_metas = [
            shard_root[self._sensor_group][self._sensor_id].attrs.asdict()
            for shard_root in self._shard_roots
        ]

        # Construct frame offset map / sensor frame time-range over all shards
        sensor_frame_timestamps_us = [
            shard_root[self._sensor_group][self._sensor_id]['frame_timestamps_us'] for shard_root in self._shard_roots
        ]

        # Offset map [0, len(s0), len(s0+s1), ... , len(s0+..+sN)]
        self._shard_frame_map = np.hstack([
            0, np.cumsum([len(shard_timestamps_us) for shard_timestamps_us in sensor_frame_timestamps_us])
        ]).astype(np.uint64)

        # Remember single sensor meta of first shard
        # [TODO(janickm): add consistency check on static data across all shards?]
        self._sensor_meta = SimpleNamespace(**shard_sensor_metas[0])

        # Global list of all frame timestamps across all shards
        self._sensor_frame_timestamps_us = np.hstack(sensor_frame_timestamps_us)

        assert len(self._sensor_frame_timestamps_us) == self._shard_frame_map[-1]

    def get_sensor_id(self) -> str:
        ''' Returns the current sensor's ID '''
        return self._sensor_id

    # Extrinsics
    @lru_cache(maxsize=1)
    def get_T_sensor_rig(self) -> np.ndarray:
        ''' Returns constant sensor-to-rig pose '''
        return np.array(self._sensor_meta.T_sensor_rig, dtype=np.float32)

    @lru_cache(maxsize=1)
    def get_T_rig_sensor(self) -> np.ndarray:
        ''' Returns constant rig-to-sensor pose '''
        return transformations.se3_inverse(self.get_T_sensor_rig())

    # Sequence-wide frame data
    def get_frames_count(self) -> int:
        ''' Returns number of frames '''
        return len(self._sensor_frame_timestamps_us)

    def get_frame_index_range(self, start_frame: int = 0, end_frame: int = -1, step_frame: int = 1) -> range:
        ''' Returns a specific range of frame indices following range(start,end,step) conventions '''

        if end_frame == -1:
            end_frame = self.get_frames_count()

        assert start_frame >= 0 and end_frame <= self.get_frames_count(), IndexError

        return range(start_frame, end_frame, step_frame)

    def get_frames_timestamps_us(self) -> np.ndarray:
        ''' Returns all end-of-measurement frame timestamps '''
        return self._sensor_frame_timestamps_us

    # Frame-dependent poses / timestamps
    @lru_cache
    def _get_frame_group(self, continous_frame_index: int) -> zarr.Group:
        ''' Returns the zarr group for a specific frame '''

        shard_frame = self._get_shard_frame(continous_frame_index)

        return self._shard_roots[shard_frame.shard_index][self._sensor_group] \
                [self._sensor_id][util.padded_index_string(shard_frame.shard_frame_index)]

    @lru_cache
    def _get_frame_meta(self, continous_frame_index: int) -> dict:
        ''' Returns frame-specific meta data '''

        return self._get_frame_group(continous_frame_index).attrs.asdict()

    @lru_cache
    def get_frame_T_rig_world(self,
                              continous_frame_index: int,
                              frame_timepoint: types.FrameTimepoint = types.FrameTimepoint.END) -> np.ndarray:
        ''' Returns start/end rig-to-world pose of specific frame '''

        return np.array(self._get_frame_meta(continous_frame_index)['T_rig_worlds'][frame_timepoint.value])

    @lru_cache
    def get_frame_T_world_rig(self,
                              continous_frame_index: int,
                              frame_timepoint: types.FrameTimepoint = types.FrameTimepoint.END) -> np.ndarray:
        ''' Returns start/end world-to-rig pose of specific frame '''
        return transformations.se3_inverse(self.get_frame_T_rig_world(continous_frame_index, frame_timepoint))

    @lru_cache
    def get_frame_T_sensor_world(self,
                                 continous_frame_index: int,
                                 frame_timepoint: types.FrameTimepoint = types.FrameTimepoint.END) -> np.ndarray:
        ''' Returns start/end sensor-to-world pose of specific frame '''

        return self.get_frame_T_rig_world(continous_frame_index, frame_timepoint) @ self.get_T_sensor_rig()

    @lru_cache
    def get_frame_T_world_sensor(self, continous_frame_index: int, frame_timepoint: types.FrameTimepoint = types.FrameTimepoint.END) -> np.ndarray:
        ''' Returns start/end world-to-sensor pose of specific frame '''
        return transformations.se3_inverse(self.get_frame_T_sensor_world(continous_frame_index, frame_timepoint))

    @lru_cache
    def get_frame_timestamp_us(self,
                               continous_frame_index: int,
                               frame_timepoint: types.FrameTimepoint = types.FrameTimepoint.END) -> int:
        ''' Returns timestamp of specific frame timepoints '''

        return self._get_frame_meta(continous_frame_index)['timestamps_us'][frame_timepoint.value]

    @lru_cache
    def get_closest_frame_index(self, timestamp_us: int) -> int:
        ''' Given a timestamp, returns the frame index of the closes frame '''

        return util.closest_index_sorted(self.get_frames_timestamps_us(), timestamp_us)


class CameraSensor(Sensor):
    ''' Provides access to camera-specific sensor-data '''

    # Image Frame Data
    class EncodedImageDataHandle():
        ''' References encoded image data without loading it '''
        def __init__(self, image_dataset: zarr.Array):
            self._image_dataset = image_dataset

        @lru_cache(maxsize=1)
        def get_data(self) -> types.EncodedImageData:
            ''' Loads the referenced encoded image data to memory '''
            return types.EncodedImageData(self._image_dataset[()], self._image_dataset.attrs['format'])

    @lru_cache(maxsize=10)
    def get_frame_handle(self, continous_frame_index: int) -> EncodedImageDataHandle:
        ''' Returns the frame's encoded image data '''
        return self.EncodedImageDataHandle(self._get_frame_group(continous_frame_index)['image'])

    @lru_cache(maxsize=10)
    def get_frame_data(self, continous_frame_index: int) -> types.EncodedImageData:
        ''' Returns the frame's encoded image data '''
        return self.get_frame_handle(continous_frame_index).get_data()

    @lru_cache(maxsize=10)
    def get_frame_image(self, continous_frame_index: int) -> PILImage.Image:
        ''' Returns the frame's decoded image data '''
        return self.get_frame_data(continous_frame_index).get_decoded_image()

    @lru_cache(maxsize=10)
    def get_frame_image_array(self, continous_frame_index: int) -> np.ndarray:
        ''' Returns decoded image data as array [W,H,C] '''
        return np.asarray(self.get_frame_image(continous_frame_index))

    # Intrinsics
    @lru_cache(maxsize=1)
    def get_camera_model_parameters(
            self) -> Union[types.FThetaCameraModelParameters, types.PinholeCameraModelParameters]:
        ''' Returns parameters specific to the camera's intrinsic model '''
        if self._sensor_meta.camera_model_type == 'ftheta':
            return types.FThetaCameraModelParameters.from_dict(self._sensor_meta.camera_model_parameters)
        if self._sensor_meta.camera_model_type == 'pinhole':
            return types.PinholeCameraModelParameters.from_dict(self._sensor_meta.camera_model_parameters)
        raise ValueError

    # Camera Mask
    @lru_cache(maxsize=1)
    def get_camera_mask_image(self) -> Optional[PILImage.Image]:
        ''' Returns constant camera mask image, if available '''

        # Take mask from *first* shard
        if (mask_dataset := self._shard_roots[0][self._sensor_group][self._sensor_id].get('mask', default=None)) is None:
            return None

        return PILImage.open(io.BytesIO(mask_dataset[()]), formats=[mask_dataset.attrs['format']])


class PointCloudSensor(Sensor):
    ''' Provides access to sensor's measuring point-clouds '''
    @lru_cache
    def has_frame_data(self, continous_frame_index: int, name: str) -> bool:
        ''' Signals if specifically named frame-data property exists '''

        return name in self._get_frame_group(continous_frame_index)

    @lru_cache(maxsize=10)
    def get_frame_data(self, continous_frame_index: int, name: str) -> np.ndarray:
        ''' Returns frame-data for a specific frame and column-name '''

        return self._get_frame_group(continous_frame_index)[name][()]


class LidarSensor(PointCloudSensor):
    ''' Provides access to lidar-specific sensor-data '''
    @lru_cache
    def get_frame_labels(self, continous_frame_index: int) -> list[types.FrameLabel3]:
        ''' Returns frame-labels for a specific frame '''

        return [
            types.FrameLabel3.from_dict(frame_label)
            for frame_label in self._get_frame_group(continous_frame_index)['frame_labels']
        ]


class RadarSensor(PointCloudSensor):
    ''' Provides access to radar-specific sensor-data '''
    pass


class ShardDataLoader:
    ''' ShardDataLoader providing convenience methods to load data generated by data-converters '''

    @staticmethod
    def evaluate_shard_file_pattern(pattern: str, skip_suffixes : list[str]=['.taridx.xz']) -> list[Path]:
        ''' Given a shard-file-pattern returns a list of matching and existing files

        Supported patterns (mutually exclusive):
        - integer-ranges: '/some/path/file-[1-3]' will be expanded to [/some/path/file-1, /some/path/file-2, /some/path/file-3]

        '''

        pattern_basepath = Path(pattern).parent
        pattern_name = Path(pattern).name

        evaluated_name_patterns = []

        # expand integer ranges like '[1-13]'
        if range_match := re.search(r'\[(\d+)-(\d+)\]', pattern_name):
            low = int(range_match.group(1))
            high = int(range_match.group(2))

            for i in range(low, high + 1):
                evaluated_name_patterns.append(pattern_name.replace(f'[{low}-{high}]', str(i) + '-'))
        else:
            evaluated_name_patterns.append(pattern_name)

        matches: set[Path] = set()
        for evaluated_pattern in evaluated_name_patterns:
            for candidate in pattern_basepath.iterdir():
                if candidate.name.startswith(evaluated_pattern):
                    skip = False
                    for skip_suffix in skip_suffixes:
                        if str(candidate).endswith(skip_suffix):
                            skip = True
                            break
                    if not skip:
                        matches.add(candidate)

        return list(matches)

    def __init__(self, shard_files: Union[list[Path], list[str]], open_consolidated: bool = True):
        assert len(shard_files), "No shard inputs provided"

        # Load shards and check for sequence consistency and continuity of shards
        shards_root_map: dict[int, zarr.Group] = {}
        for f in shard_files:
            
            logging.info(f'ShardDataLoader: Loading shard file {f}')

            timer = common.SimpleTimer()
            store = IndexedTarStore(f, mode='r')
            if open_consolidated:
                shard_root = zarr.open_consolidated(store=store, mode='r')
            else:
                shard_root = zarr.open(store=store, mode='r')

            logging.info(f'ShardDataLoader: time_load={timer.elapsed_sec(restart = True)}sec | open_consolidated={open_consolidated}')

            shard_sequence_id = shard_root.attrs.get('sequence_id')
            shard_camera_ids = set(shard_root.attrs.get('camera_ids'))
            shard_lidar_ids = set(shard_root.attrs.get('lidar_ids'))
            shard_radar_ids = set(shard_root.attrs.get('radar_ids'))
            shard_shard_id = shard_root.attrs.get('shard_id')
            shard_shard_count = shard_root.attrs.get('shard_count')
            shard_shard_version = shard_root.attrs.get('version')

            if not shards_root_map:
                self._sequence_id: str = shard_sequence_id
                self._camera_ids: set[str] = shard_camera_ids
                self._lidar_ids: set[str] = shard_lidar_ids
                self._radar_ids: set[str] = shard_radar_ids
                self._shard_count: int = shard_shard_count
                self._shard_version: str = shard_shard_version

            assert self._sequence_id == shard_sequence_id, "Can't load shards from different sequences"
            assert self._camera_ids == shard_camera_ids, "Can't load shards with different camera sensors"
            assert self._lidar_ids == shard_lidar_ids, "Can't load shards with different lidar sensors"
            assert self._radar_ids == shard_radar_ids, "Can't load shards with different radar sensors"
            assert self._shard_count == shard_shard_count, "Can't load shards from different subdivisions"
            assert self._shard_version == shard_shard_version, "Can't load shards from different data versions"

            assert shard_shard_id not in shards_root_map, "Shard ID loaded multiple times"
            shards_root_map[shard_shard_id] = shard_root

        # Check version-compatibility
        assert self._shard_version == VERSION, 'loading incompatible version'  # TODO: this check can still be refined

        # Make sure shard IDs are continous
        self._shard_ids = sorted(list(shards_root_map.keys()))
        assert self._shard_ids[-1] - self._shard_ids[0] + 1 == len(self._shard_ids), f"Non-continous sequence of shards: {self._shard_ids}"

        # *Linear* sequence of shard files
        self._shard_files = [shards_root_map[shard_id] for shard_id in self._shard_ids]

    @lru_cache(maxsize=1)
    def get_poses(self) -> types.Poses:
        ''' Returns all timestamped poses associated with the dataset '''

        # Load common base pose
        # [TODO(janickm): add consistency check on static data across all shards?]
        T_rig_world_base = self._shard_files[0]['poses']['T_rig_world_base']

        # Concat all poses from all shards
        T_rig_worlds = []
        T_rig_world_timestamps_us = []
        for shard_file in self._shard_files:
            T_rig_worlds.append(shard_file['poses']['T_rig_worlds'])
            T_rig_world_timestamps_us.append(shard_file['poses']['T_rig_world_timestamps_us'])

        # [TODO(janickm): add timestamp uniqueness check on data across all shards?]
        return types.Poses(np.array(T_rig_world_base), np.vstack(T_rig_worlds), np.hstack(T_rig_world_timestamps_us))

    def get_sequence_id(self, with_shard_range: bool) -> str:
        ''' Provides access to a unique identifier of the loaded shard data, optionally including the linear range of shards

            Examples:
            - with_shard_range == False: c9b05cf4-afb9-11ec-b3c2-00044bf65fcb
            - with_shard_range == True:  c9b05cf4-afb9-11ec-b3c2-00044bf65fcb_2_3_4 [assuming shards 2,3,4 were loaded]
        '''
        if with_shard_range:
            return f"{self._sequence_id}_{'_'.join([str(shard_id) for shard_id in self._shard_ids])}"
        return self._sequence_id

    def get_sensor(self, sensor_id: str) -> Union[CameraSensor, LidarSensor, RadarSensor]:
        ''' Provides access to a specific sensor given it's sensor-id '''
        if sensor_id in self._camera_ids:
            return self.get_camera_sensor(sensor_id)
        if sensor_id in self._lidar_ids:
            return self.get_lidar_sensor(sensor_id)
        if sensor_id in self._radar_ids:
            return self.get_radar_sensor(sensor_id)
        raise ValueError(f'Unknown sensor {sensor_id}')

    @cache
    def get_camera_sensor(self, camera_id) -> CameraSensor:
        ''' Provides access to a specific camera sensor given it's sensor-id '''
        return CameraSensor(camera_id, CAMERAS_BASE_GROUP, self._shard_files)

    def get_camera_ids(self) -> list[str]:
        ''' Returns all camera sensor ids '''
        return list(self._camera_ids)

    @cache
    def get_lidar_sensor(self, lidar_id) -> LidarSensor:
        ''' Provides access to a specific lidar sensor given it's sensor-id '''
        return LidarSensor(lidar_id, LIDARS_BASE_GROUP, self._shard_files)

    def get_lidar_ids(self) -> list[str]:
        ''' Returns all lidar sensor ids '''
        return list(self._lidar_ids)

    @cache
    def get_radar_sensor(self, radar_id) -> RadarSensor:
        ''' Provides access to a specific radar sensor given it's sensor-id '''
        return RadarSensor(radar_id, RADARS_BASE_GROUP, self._shard_files)

    def get_radar_ids(self) -> list[str]:
        ''' Returns all radar sensor ids '''
        return list(self._radar_ids)
