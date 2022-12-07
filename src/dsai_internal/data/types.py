# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from enum import IntEnum, auto, unique
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import dataclasses_json

from . import util

## Data classes representing stored data types
@unique
class ShutterType(IntEnum):
    ''' Enumerates different possible shutter types '''
    ROLLING_TOP_TO_BOTTOM = auto()
    ROLLING_LEFT_TO_RIGHT = auto()
    ROLLING_BOTTOM_TO_TOP = auto()
    ROLLING_RIGHT_TO_LEFT = auto()
    GLOBAL = auto()

@dataclass
class CameraModelParameters:
    ''' Represents parameters common to all camera models '''
    resolution: np.ndarray = util.numpy_array_field(np.uint64)
    shutter_type: ShutterType = util.enum_field(ShutterType)
    exposure_time_us: int = 0

    def __post_init__(self):
        # Sanity checks
        assert self.resolution.shape == (2, )
        assert self.resolution.dtype == np.dtype('uint64')
        assert self.resolution[0] > 0 and self.resolution[1] > 0
        assert self.exposure_time_us > 0


@dataclass
class FThetaCameraModelParameters(CameraModelParameters, dataclasses_json.DataClassJsonMixin):
    ''' Represents FTheta-specific camera model parameters '''
    principal_point: np.ndarray = util.numpy_array_field(np.float32)
    bw_poly: np.ndarray = util.numpy_array_field(np.float32)
    fw_poly: np.ndarray = util.numpy_array_field(np.float32)
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
class PinholeCameraModelParameters(CameraModelParameters, dataclasses_json.DataClassJsonMixin):
    ''' Represents a Pinhole-specific camera model parameters '''
    principal_point: np.ndarray = util.numpy_array_field(np.float32)
    focal_length_u: float = 0.0
    focal_length_v: float = 0.0
    radial_poly: np.ndarray = util.numpy_array_field(np.float32) 
    tangential_poly: np.ndarray = util.numpy_array_field(np.float32)
    #TODO: do we also want to add the thin prism distortion coefficients?

    @staticmethod
    def type() -> str:
        return 'pinhole'

    def __post_init__(self):
        # Sanity checks
        super().__post_init__()
        assert self.principal_point.shape == (2, )
        assert self.principal_point.dtype == np.dtype('float32')
        assert self.principal_point[0] > 0.0 and self.principal_point[1] > 0.0

        assert self.radial_poly.shape == (6,)
        assert self.radial_poly.dtype == np.dtype('float32')

        assert self.tangential_poly.shape == (2, )
        assert self.tangential_poly.dtype == np.dtype('float32')

        assert self.focal_length_u > 0
        assert self.focal_length_v > 0


@dataclass
class Poses:
    ''' Represents a collection of timestamped poses (rig-to-local-world transformation) '''
    T_rig_world_base: np.ndarray
    T_rig_worlds: np.ndarray
    T_rig_world_timestamps_us: np.ndarray

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

    def to_array(self) -> np.ndarray:
        ''' Convenience single-array representation '''
        return np.array(self.centroid + self.dim + self.rot, dtype=np.float32)

@unique
class LabelSource(IntEnum):
    ''' Enumerates different sources for labels (auto, manual, GT, synthetic etc.) '''
    AUTOLABEL = auto()


@dataclass
class FrameLabel3(dataclasses_json.DataClassJsonMixin):
    ''' Description of a 3D frame-associated label '''
    label_id: str
    track_id: str
    label_class: str
    bbox3: BBox3
    global_speed: float
    confidence: float
    source: LabelSource = util.enum_field(LabelSource)


@dataclass
class TrackLabel(dataclasses_json.DataClassJsonMixin):
    ''' Description of an object-specific track '''
    dynamic_flag: bool
    sensors: dict[str, list[int]]  # all frame-timestamps of the object in different sensors


@unique
class DynamicFlagState(IntEnum):
    ''' Enumerates potential per-point flag values related to 'dynamic_flag' property '''
    NOT_AVAILABLE = -1
    STATIC = 0
    DYNAMIC = 1


@unique
class FrameTimepoint(IntEnum):
    ''' Enumerates special timepoints within a frame (values used to index into buffers) '''
    START = 0
    END = 1
