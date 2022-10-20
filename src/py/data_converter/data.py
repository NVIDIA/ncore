# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from enum import Enum, auto
import json
import shutil
from types import SimpleNamespace

import h5py

import numpy as np
from PIL.Image import Image

from pathlib import Path
from typing import Optional, Tuple, Union

from dataclasses import dataclass, field, Field
import dataclasses_json
import marshmallow

## Constants
VERSION = '2.0.0'
INDEX_DIGITS = 6  # the number of integer digits to pad counters in output filenames
CAMERAS_BASE_DIR = 'cameras'
LIDARS_BASE_DIR = 'lidars'
RADARS_BASE_DIR = 'radars'


## Helper types and functions
class FrameTimepoint(Enum):
    ''' Enumerates special timepoints within a frame '''
    START = 0
    END = 1


def padded_index_string(index: int, index_digits=INDEX_DIGITS) -> str:
    ''' Pads an integer with leading zeros to a fixed number of digits '''
    return str(index).zfill(index_digits)


def numpy_array_field(datatype: np.dtype, default=None):
    ''' Provides encoder / decoder functionality for numpy arrays into dataclass JSON-compatible types'''
    def decoder(*args, **kwargs):
        return np.array(*args, dtype=datatype, **kwargs)

    return field(default=default,
                 metadata=dataclasses_json.config(encoder=np.ndarray.tolist,
                                                  decoder=decoder,
                                                  mm_field=marshmallow.fields.List))


def enum_field(enum_class, default=None):
    def encoder(v):
        return v.name

    def decoder(*args, **kwargs):
        return enum_class(*args, **kwargs)

    return field(default=default,
                 metadata=dataclasses_json.config(encoder=encoder, decoder=decoder, mm_field=marshmallow.fields.Enum))


class AutoNameEnum(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


## Data classes representing stored data types
class ShutterType(AutoNameEnum):
    ''' Enumerates different possible shutter types '''
    ROLLING_TOP_TO_BOTTOM = auto()
    ROLLING_LEFT_TO_RIGHT = auto()
    ROLLING_BOTTOM_TO_TOP = auto()
    ROLLING_RIGHT_TO_LEFT = auto()
    GLOBAL = auto()

@dataclass
class CameraModel:
    ''' Represents properties common to all camera models '''
    resolution: np.array = numpy_array_field(np.uint64)
    shutter_type: ShutterType = enum_field(ShutterType)
    exposure_time_us: int = 0

    def __post_init__(self):
        # Sanity checks
        assert self.resolution.shape == (2, )
        assert self.resolution.dtype == np.dtype('uint64')
        assert self.resolution[0] > 0 and self.resolution[1] > 0
        assert self.exposure_time_us > 0


@dataclass
class FThetaCameraModel(CameraModel, dataclasses_json.DataClassJsonMixin):
    ''' Represents FTheta-specific camera model parameters '''
    principal_point: np.array = numpy_array_field(np.float32)
    bw_poly: np.array = numpy_array_field(np.float32)
    fw_poly: np.array = numpy_array_field(np.float32)
    max_angle: float = 0.0

    @staticmethod
    def type() -> str:
        return 'ftheta'

    POLYNOMIAL_DEGREE = 6

    def __post_init__(self):
        # Sanity checks
        super().__post_init__()
        assert self.principal_point.shape == (2, )
        assert self.principal_point.dtype == np.dtype('float32')
        assert self.principal_point[0] > 0.0 and self.principal_point[1] > 0.0

        assert self.bw_poly.ndim == 1
        assert len(self.bw_poly) <= self.POLYNOMIAL_DEGREE
        assert self.bw_poly.dtype == np.dtype('float32')

        assert self.fw_poly.ndim == 1
        assert len(self.fw_poly) <= self.POLYNOMIAL_DEGREE
        assert self.fw_poly.dtype == np.dtype('float32')

        # pad polynomials to full size
        self.bw_poly = np.pad(self.bw_poly, (0,self.POLYNOMIAL_DEGREE - len(self.bw_poly)), mode='constant', constant_values=0.0)
        self.fw_poly = np.pad(self.fw_poly, (0,self.POLYNOMIAL_DEGREE - len(self.fw_poly)), mode='constant', constant_values=0.0)

        assert self.max_angle > 0.0


@dataclass
class PinholeCameraModel(CameraModel, dataclasses_json.DataClassJsonMixin):
    ''' Represents a Pinhole-specific camera model parameters '''
    principal_point: np.array = numpy_array_field(np.float32)
    focal_length_u: float = 0.0
    focal_length_v: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0

    @staticmethod
    def type() -> str:
        return 'pinhole'

    def __post_init__(self):
        # Sanity checks
        super().__post_init__()
        assert self.principal_point.shape == (2, )
        assert self.principal_point.dtype == np.dtype('float32')
        assert self.principal_point[0] > 0.0 and self.principal_point[1] > 0.0

        assert self.focal_length_u > 0
        assert self.focal_length_v > 0


@dataclass
class Poses:
    ''' Represents a collection of timestamped poses (rig-to-local-world transformation) '''
    T_rig_world_base: np.array
    T_rig_worlds: np.array
    T_rig_world_timestamps_us: np.array

    def __post_init__(self):
        # Sanity checks
        assert self.T_rig_world_base.shape == (4, 4)
        assert self.T_rig_world_base.dtype == np.dtype('float64')

        assert self.T_rig_worlds.shape[1:] == (4, 4)
        assert self.T_rig_worlds.dtype == np.dtype('float64')

        assert self.T_rig_world_timestamps_us.ndim == 1
        assert self.T_rig_world_timestamps_us.dtype == np.dtype('uint64')

        assert self.T_rig_worlds.shape[0] == self.T_rig_world_timestamps_us.shape[0]


@dataclass
class BBox3(dataclasses_json.DataClassJsonMixin):
    ''' Parameters of a 3D bounding-box '''
    centroid: Tuple[float, float, float]
    dim: Tuple[float, float, float]
    rot: Tuple[float, float, float]

    def to_array(self) -> np.array:
        ''' Convenience single-array representation '''
        return np.array(self.centroid + self.dim + self.rot, dtype=np.float32)


class LabelSource(str, Enum):
    ''' Enumerates different sources for labels (auto, manual, GT, synthetic etc.) '''
    AUTOLABEL = 'autolabel'


@dataclass
class FrameLabel3(dataclasses_json.DataClassJsonMixin):
    ''' Description of a 3D frame-associated label '''
    label_id: str
    track_id: str
    label_class: str
    source: LabelSource
    bbox3: BBox3
    global_speed: float
    confidence: float


@dataclass
class TrackLabel(dataclasses_json.DataClassJsonMixin):
    ''' Description of an object-specific track '''
    dynamic_flag: bool
    sensors: dict[str, list[int]]  # all frame-timestamps of the object in different sensors


class DynamicFlagState(Enum):
    ''' Enumerates potential per-point flag values related to 'dynamic_flag' property '''
    NOT_AVAILABLE = -1
    STATIC = 0
    DYNAMIC = 1


class DataWriter():
    ''' DataWriter implementing format-specific serialization specifications to store data generated by data-converters '''
    def __init__(self, sequence_output_dir: Path, camera_ids: list[str], lidar_ids: list[str], radar_ids: list[str]):
        ''' 
        Instantiate data writer and initialize the default folder structure for a given sequence and sensor IDs
        '''

        self.camera_ids = camera_ids
        self.lidar_ids = lidar_ids
        self.radar_ids = radar_ids

        # Check that sensor-id's are unique across sensor types
        assert len(set(self.camera_ids + self.lidar_ids + self.radar_ids)) == len(self.camera_ids) + len(
            self.lidar_ids) + len(self.radar_ids), 'sensor id\'s are not unique'

        self.sequence_output_dir = sequence_output_dir

        # Create output folder structure
        for camera_id in self.camera_ids:
            (self.sequence_output_dir / CAMERAS_BASE_DIR / camera_id).mkdir(parents=True, exist_ok=True)

        for lidar_id in self.lidar_ids:
            (self.sequence_output_dir / LIDARS_BASE_DIR / lidar_id).mkdir(parents=True, exist_ok=True)

        for radar_id in self.radar_ids:
            (self.sequence_output_dir / RADARS_BASE_DIR / radar_id).mkdir(parents=True, exist_ok=True)

    # Individual 'store*' methods performing data sanity checks and serialize consistent output formats
    def store_poses(self, poses: Poses) -> None:
        with h5py.File(self.sequence_output_dir / 'poses.hdf5', "w") as f:
            f.create_dataset('T_rig_world_base', data=poses.T_rig_world_base)
            f.create_dataset('T_rig_worlds', data=poses.T_rig_worlds)
            f.create_dataset('T_rig_world_timestamps_us', data=poses.T_rig_world_timestamps_us)

    def store_labels(self, track_labels: dict[str, TrackLabel]) -> None:
        output = {'track_labels': {k: v.to_dict() for (k, v) in track_labels.items()}}

        # TODO: add sanity checks on the final label structure before output

        with open(self.sequence_output_dir / 'labels.json', "w") as outfile:
            outfile.write(json.dumps(output))

    def store_camera_frame(
            self,

            # data indexing
            camera_id: str,
            continous_frame_index: int,

            # image data
            source_image_path: Path,
            symlink_frame: bool,

            # poses
            T_rig_worlds: np.array,
            timestamps_us: np.array) -> None:
        # sanity / consistency checks
        assert T_rig_worlds.shape == (2, 4, 4)
        assert T_rig_worlds.dtype == np.dtype('float32')
        assert timestamps_us.shape == (2, )
        assert timestamps_us.dtype == np.dtype('uint64')

        sensor_output_dir = self.sequence_output_dir / CAMERAS_BASE_DIR / camera_id
        continous_frame_index_string = padded_index_string(continous_frame_index)

        # Copy / symlink image from source to target
        target_image_path = sensor_output_dir / (continous_frame_index_string + '.jpeg')

        if symlink_frame:
            # Create symlink target -> source
            Path(target_image_path).symlink_to(source_image_path)
        else:
            # Perform explicit frame file copy
            shutil.copy(source_image_path, target_image_path)

        # Output frame meta data
        output = {'T_rig_worlds': T_rig_worlds.tolist(), 'timestamps_us': timestamps_us.tolist()}

        with open(sensor_output_dir / (continous_frame_index_string + '.json'), 'w') as outfile:
            outfile.write(json.dumps(output))

    def store_camera_meta(
            self,
            camera_id: str,

            # all frame timestamps
            frame_timestamps_us: np.array,

            # extrinsics
            T_sensor_rig: np.array,

            # intrinsics
            camera_model: Union[FThetaCameraModel, PinholeCameraModel],

            # sensor constants
            mask_image: Optional[Image]) -> None:
        assert T_sensor_rig.shape == (4, 4)
        assert T_sensor_rig.dtype == np.dtype('float32')
        assert frame_timestamps_us.ndim == 1
        assert frame_timestamps_us.dtype == np.dtype('uint64')

        sensor_output_dir = self.sequence_output_dir / CAMERAS_BASE_DIR / camera_id

        output = {
            'frame_timestamps_us': frame_timestamps_us.tolist(),
            'T_sensor_rig': T_sensor_rig.tolist(),
            'camera_model_type': camera_model.type(),
            'camera_model': camera_model.to_dict()
        }

        with open(sensor_output_dir / 'meta.json', 'w') as outfile:
            outfile.write(json.dumps(output))

        # Output mask if available
        if mask_image:
            mask_image.save(sensor_output_dir / 'mask.png', optimize=True)

    def store_lidar_frame(
            self,

            # data indexing
            lidar_id: str,
            continous_frame_index: int,

            # point-cloud data
            xyz_s: np.array,
            xyz_e: np.array,
            intensity: np.array,
            timestamp_us: np.array,
            dynamic_flag: np.array,

            # label data
            frame_labels: list[FrameLabel3],

            # poses
            T_rig_worlds: np.array,
            timestamps_us: np.array) -> None:
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

        sensor_output_dir = self.sequence_output_dir / LIDARS_BASE_DIR / lidar_id
        continous_frame_index_string = padded_index_string(continous_frame_index)

        # Store frame data
        target_frame_path = sensor_output_dir / (continous_frame_index_string + '.hdf5')
        with h5py.File(target_frame_path, "w") as f:
            COMPRESSION = 'lzf'
            f.create_dataset('xyz_s', data=xyz_s, compression=COMPRESSION)
            f.create_dataset('xyz_e', data=xyz_e, compression=COMPRESSION)
            f.create_dataset('intensity', data=intensity, compression=COMPRESSION)
            f.create_dataset('timestamp', data=timestamp_us, compression=COMPRESSION)
            f.create_dataset('dynamic_flag', data=dynamic_flag, compression=COMPRESSION)

        # Output frame meta data
        output = {
            'T_rig_worlds': T_rig_worlds.tolist(),
            'timestamps_us': timestamps_us.tolist(),
            'frame_labels': [frame_label.to_dict() for frame_label in frame_labels]
        }

        with open(sensor_output_dir / (continous_frame_index_string + '.json'), 'w') as outfile:
            outfile.write(json.dumps(output))

    def store_lidar_meta(
            self,
            lidar_id: str,

            # all frame timestamps
            frame_timestamps_us: np.array,

            # extrinsics
            T_sensor_rig: np.array) -> None:
        assert T_sensor_rig.shape == (4, 4)
        assert T_sensor_rig.dtype == np.dtype('float32')
        assert frame_timestamps_us.shape[1:] == ()
        assert frame_timestamps_us.dtype == np.dtype('uint64')

        sensor_output_dir = self.sequence_output_dir / LIDARS_BASE_DIR / lidar_id

        output = {'frame_timestamps_us': frame_timestamps_us.tolist(), 'T_sensor_rig': T_sensor_rig.tolist()}

        with open(sensor_output_dir / 'meta.json', 'w') as outfile:
            outfile.write(json.dumps(output))

    def store_meta(self, calibration_type: str, egomotion_type: str) -> None:
        # Store dataset associated meta-data

        output = {
            'version': VERSION,
            'egomotion_type': egomotion_type,
            'calibration_type': calibration_type,
            'sensors': {
                'lidars': self.lidar_ids,
                'cameras': self.camera_ids,
                'radars': self.radar_ids,
            }
        }

        with open(self.sequence_output_dir / 'meta.json', 'w') as outfile:
            outfile.write(json.dumps(output))

    def store_shard_meta(self, shard_id: int, shard_count: int, successful: bool) -> None:
        with open(self.sequence_output_dir / f'shard-meta-{padded_index_string(shard_id, index_digits=4)}.json', 'w') as outfile:
            json.dump({'shard-id': shard_id, 'shard-count': shard_count, 'successful': successful}, outfile)

class Sensor:
    ''' Provides access to generic data available to all sensor types '''
    def __init__(self, sensor_id: str, sensor_dir: Path):
        self._sensor_id = sensor_id
        self._sensor_dir = sensor_dir

        # load meta-data
        with open(self._sensor_dir / 'meta.json', 'r') as f:
            self._sensor_meta = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    def get_sensor_id(self) -> str:
        ''' Returns the current sensor's ID '''
        return self._sensor_id

    def get_sensor_dir(self) -> Path:
        ''' Returns the current sensor's data folder '''
        return self._sensor_dir

    # Extrinsics
    def get_T_sensor_rig(self) -> np.array:
        ''' Returns constant sensor-to-rig pose '''
        return np.array(self._sensor_meta.T_sensor_rig, dtype=np.float32)

    # Sessions-wide frame data
    def get_frames_count(self) -> int:
        ''' Returns number of frames '''
        return len(self._sensor_meta.frame_timestamps_us)

    def get_frame_index_range(self, start_frame :int = 0, end_frame : int = -1, step_frame : int = 1) -> range:
        ''' Returns a specific range of frame indices following range(start,end,step) conventions '''

        if end_frame == -1:
            end_frame = self.get_frames_count()

        assert start_frame >= 0 and end_frame <= self.get_frames_count(), IndexError

        return range(start_frame, end_frame, step_frame)

    def get_frames_timestamps_us(self) -> np.array:
        ''' Returns all end-of-measurement frame timestamps '''
        return np.array(self._sensor_meta.frame_timestamps_us, dtype=np.uint64)

    # Frame-dependent poses / timestamps
    def get_frame_T_rig_world(self, continous_frame_index: int, frame_timepoint: FrameTimepoint = FrameTimepoint.END) -> np.array:
        ''' Returns start/end rig-to-world pose of specific frame '''

        with open(self._sensor_dir / (padded_index_string(continous_frame_index) + '.json'), 'r') as f:
            j = json.load(f)
            return np.array(j['T_rig_worlds'][frame_timepoint.value])

    def get_frame_T_sensor_world(self, continous_frame_index: int, frame_timepoint: FrameTimepoint = FrameTimepoint.END) -> np.array:
        ''' Returns start/end sensor-to-world pose of specific frame '''

        with open(self._sensor_dir / (padded_index_string(continous_frame_index) + '.json'), 'r') as f:
            j = json.load(f)
            return np.array(j['T_rig_worlds'][frame_timepoint.value]) @ self.get_T_sensor_rig()

    def get_frame_timestamp_us(self, continous_frame_index: int, frame_timepoint: FrameTimepoint = FrameTimepoint.END) -> int:
        ''' Returns timestamp of specific frame timepoints '''

        with open(self._sensor_dir / (padded_index_string(continous_frame_index) + '.json'), 'r') as f:
            j = json.load(f)
            return j['timestamps_us'][frame_timepoint.value]


class CameraSensor(Sensor):
    ''' Provides access to camera-specific sensor-data '''
    def get_camera_model(self) -> Union[FThetaCameraModel, PinholeCameraModel]:
        if self._sensor_meta.camera_model_type == 'ftheta':
            return FThetaCameraModel.from_dict(self._sensor_meta.camera_model.__dict__)
        if self._sensor_meta.camera_model_type == 'pinhole':
            return PinholeCameraModel.from_dict(self._sensor_meta.camera_model.__dict__)
        raise ValueError

    def get_frame_image_path(self, continous_frame_index: int) -> Path:
        ''' Returns the path to the image file of a specific frame '''
        assert continous_frame_index >= 0 and continous_frame_index < self.get_frames_count(), IndexError

        return self.get_sensor_dir() / (padded_index_string(continous_frame_index) + '.jpeg')


class PointCloudSensor(Sensor):
    ''' Provides access to sensor's measureing point-clouds '''
    def get_frame_data(self, continous_frame_index: int, key: str) -> np.array:
        ''' Returns frame-data for a specific frame and column-key '''
        assert continous_frame_index >= 0 and continous_frame_index < self.get_frames_count(), IndexError

        return np.array(
            h5py.File(self.get_sensor_dir() / (padded_index_string(continous_frame_index) + '.hdf5'), 'r')[key])


class LidarSensor(PointCloudSensor):
    ''' Provides access to lidar-specific sensor-data '''
    def get_frame_labels(self, continous_frame_index: int) -> list[FrameLabel3]:
        ''' Returns frame-labels for a specific frame '''

        with open(self._sensor_dir / (padded_index_string(continous_frame_index) + '.json'), 'r') as f:
            j = json.load(f)
            return [FrameLabel3.from_dict(frame_label) for frame_label in j['frame_labels']]


class RadarSensor(PointCloudSensor):
    ''' Provides access to radar-specific sensor-data '''
    pass


class DataLoader():
    ''' DataLoader providing convenience methods to load data generated by data-converters '''
    def __init__(self, sequence_dir: Union[Path, str]):

        if isinstance(sequence_dir, str):
            sequence_dir = Path(sequence_dir)

        self.sequence_dir = sequence_dir

        # load sequence meta-data
        with open(self.sequence_dir / 'meta.json', 'r') as f:
            self._meta = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

        # check version-compatibility
        assert self._meta.version == VERSION, 'loading incompatible version'  # TODO: this check can still be refined

        # load all available sensors
        self._sensors: dict[str, Union[CameraSensor, LidarSensor, RadarSensor]] = {}
        self._sensors.update({
            sensor_id: CameraSensor(sensor_id, self.sequence_dir / CAMERAS_BASE_DIR / sensor_id)
            for sensor_id in self._meta.sensors.cameras
        })
        self._sensors.update({
            sensor_id: LidarSensor(sensor_id, self.sequence_dir / LIDARS_BASE_DIR / sensor_id)
            for sensor_id in self._meta.sensors.lidars
        })
        self._sensors.update({
            sensor_id: RadarSensor(sensor_id, self.sequence_dir / RADARS_BASE_DIR / sensor_id)
            for sensor_id in self._meta.sensors.radars
        })

        self._poses: Optional[Poses] = None  # poses will be loaded on-demand if required

    def get_poses(self) -> Poses:
        ''' Returns all timestamped poses associated with the dataset '''
        # Load poses on-demand
        if self._poses is None:
            with h5py.File(self.sequence_dir / 'poses.hdf5', 'r') as f:
                self._poses = Poses(np.array(f['T_rig_world_base']), np.array(f['T_rig_worlds']),
                                    np.array(f['T_rig_world_timestamps_us']))
        return self._poses

    def get_sensor(self, sensor_id: str) -> Union[CameraSensor, LidarSensor, RadarSensor]:
        ''' Provides access to a specific sensor given it's sensor-id '''
        return self._sensors[sensor_id]

    def get_camera_sensor_ids(self) -> list[str]:
        ''' Returns all camera sensor-ids '''
        return self._meta.sensors.cameras

    def get_lidar_sensor_ids(self) -> list[str]:
        ''' Returns all lidar sensor-ids '''
        return self._meta.sensors.lidars

    def get_radar_sensor_ids(self) -> list[str]:
        ''' Returns all radar sensor-ids '''
        return self._meta.sensors.radars
