# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import io

from enum import IntEnum, auto, unique
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple
from functools import lru_cache

import numpy as np
import dataclasses_json
import PIL.Image as PILImage

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
    @unique
    class PolynomialType(IntEnum):
        ''' Enumerates different possible polynomial types '''
        PIXELDIST_TO_ANGLE = auto()  # also known as "backward"
        ANGLE_TO_PIXELDIST = auto()  # also known as "forward"

    principal_point: np.ndarray = util.numpy_array_field(np.float32)
    reference_poly: PolynomialType = util.enum_field(PolynomialType)
    pixeldist_to_angle_poly: np.ndarray = util.numpy_array_field(np.float32)
    angle_to_pixeldist_poly: np.ndarray = util.numpy_array_field(np.float32)
    max_angle: float = 0.0

    @staticmethod
    def type() -> str:
        return 'ftheta'

    # Aliases for polynomial members
    @property
    def bw_poly(self):
        return self.pixeldist_to_angle_poly

    @property
    def fw_poly(self):
        return self.angle_to_pixeldist_poly

    POLYNOMIAL_DEGREE = 6

    def __post_init__(self):
        # Sanity checks
        super().__post_init__()
        assert self.principal_point.shape == (2, )
        assert self.principal_point.dtype == np.dtype('float32')
        assert self.principal_point[0] >= 0.0 and self.principal_point[1] >= 0.0

        assert self.pixeldist_to_angle_poly.ndim == 1
        assert len(self.pixeldist_to_angle_poly) <= self.POLYNOMIAL_DEGREE
        assert self.pixeldist_to_angle_poly.dtype == np.dtype('float32')

        assert self.angle_to_pixeldist_poly.ndim == 1
        assert len(self.angle_to_pixeldist_poly) <= self.POLYNOMIAL_DEGREE
        assert self.angle_to_pixeldist_poly.dtype == np.dtype('float32')

        # pad polynomials to full size
        self.pixeldist_to_angle_poly = np.pad(self.pixeldist_to_angle_poly, (0,self.POLYNOMIAL_DEGREE - len(self.pixeldist_to_angle_poly)), mode='constant', constant_values=0.0)
        self.angle_to_pixeldist_poly = np.pad(self.angle_to_pixeldist_poly, (0,self.POLYNOMIAL_DEGREE - len(self.angle_to_pixeldist_poly)), mode='constant', constant_values=0.0)

        assert self.max_angle > 0.0


@dataclass
class PinholeCameraModelParameters(CameraModelParameters, dataclasses_json.DataClassJsonMixin):
    ''' Represents a Pinhole-specific camera model parameters '''
    principal_point: np.ndarray = util.numpy_array_field(np.float32)
    focal_length: np.ndarray = util.numpy_array_field(np.float32)
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

        assert self.focal_length.shape == (2, )
        assert self.focal_length.dtype == np.dtype('float32')
        assert self.focal_length[0] > 0.0 and self.focal_length[1] > 0.0

        assert self.radial_poly.shape == (6,)
        assert self.radial_poly.dtype == np.dtype('float32')

        assert self.tangential_poly.shape == (2, )
        assert self.tangential_poly.dtype == np.dtype('float32')

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

    # If available, the timestamp associated with the centroid of the label
    # (possibly an accurate in-spin time). Optional also to be
    # backwards-compatible with existing datasets that don't provide
    # this information.
    #
    # In the future this field might become mandatory (deprecating old datasets)
    timestamp_us: Optional[int]
    
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


class EncodedImageData():
    ''' Represents encoded image data of a specific format in memory '''
    def __init__(self, encoded_image_data: bytes, encoded_image_format: str):
        self._encoded_image_data = encoded_image_data
        self._encoded_image_format = encoded_image_format

    def get_encoded_image_data(self) -> bytes:
        ''' Returns encoded image data '''
        return self._encoded_image_data

    def get_encoded_image_format(self) -> str:
        ''' Returns encoded image format '''
        return self._encoded_image_format

    @lru_cache(maxsize=1)
    def get_decoded_image(self) -> PILImage.Image:
        ''' Returns decoded image from image data '''
        return PILImage.open(io.BytesIO(self.get_encoded_image_data()), formats=[self.get_encoded_image_format()])


class EncodedImageHandle(Protocol):
    ''' Protocol type to reference encoded image data (e.g., file-based, container-based, memory-based) '''
    def get_data(self) -> EncodedImageData:
        ...
