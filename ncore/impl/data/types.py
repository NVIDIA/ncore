# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import io
import sys
import dataclasses

from enum import IntEnum, auto, unique
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, Union, List, Dict
from functools import lru_cache

import numpy as np
import dataclasses_json
import PIL.Image as PILImage

from ncore.impl.data import util


## Data classes representing stored data types
@unique
class ShutterType(IntEnum):
    """Enumerates different possible camera imager shutter types"""

    ROLLING_TOP_TO_BOTTOM = auto()  #: Rolling shutter from top to bottom of the imager
    ROLLING_LEFT_TO_RIGHT = auto()  #: Rolling shutter from left to right of the imager
    ROLLING_BOTTOM_TO_TOP = auto()  #: Rolling shutter from bottom to top of the imager
    ROLLING_RIGHT_TO_LEFT = auto()  #: Rolling shutter from right to left of the imager
    GLOBAL = auto()  #: Instantaneous global shutter (no rolling shutter)


@dataclass
class CameraModelParameters:
    """Represents parameters common to all camera models"""

    resolution: np.ndarray = util.numpy_array_field(
        np.uint64
    )  #: Width and height of the image in pixels (uint32, [2,])
    shutter_type: ShutterType = util.enum_field(ShutterType)  #: Shutter type of the camera's imaging sensor

    def __post_init__(self):
        # Sanity checks
        assert self.resolution.shape == (2,)
        assert self.resolution.dtype == np.dtype("uint64")
        assert self.resolution[0] > 0 and self.resolution[1] > 0


@dataclass
class FThetaCameraModelParameters(CameraModelParameters, dataclasses_json.DataClassJsonMixin):
    """Represents FTheta-specific camera model parameters"""

    @unique
    class PolynomialType(IntEnum):
        """Enumerates different possible polynomial types"""

        PIXELDIST_TO_ANGLE = (
            auto()
        )  #: Polynomial mapping pixeldistances-to-angles (also known as "backward" polynomial)
        ANGLE_TO_PIXELDIST = auto()  #: Polynomial mapping angles-to-pixeldistances (also known as "forward" polynomial)

    principal_point: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: U and v coordinate of the principal point, following the NVIDIA default convention for FTheta camera models in which the pixel indices represent the center of the pixel (not the top-left corners). Principal point coordinates will be adapted internally in camera model APIs to reflect the :ref:`image coordinate conventions <image_coordinate_conventions>`
    reference_poly: PolynomialType = util.enum_field(
        PolynomialType
    )  #: Indicating which of the two stored polynomials is the model's *reference* polynomial (the other polynomial is only an approximation)
    pixeldist_to_angle_poly: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: Coefficients of the pixeldistances-to-angles polynomial (float32, [6,])
    angle_to_pixeldist_poly: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: Coefficients of the angles-to-pixeldistances polynomial (float32, [6,])
    max_angle: float = 0.0  #: Maximal extrinsic ray angle [rad] with the principal direction (float32)
    linear_cde: np.ndarray = util.numpy_array_field(
        np.float32, default=np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )  #: Coefficients of the constrained linear term [c,d;e,1] transforming between sensor coordinates (in mm) to image coordinates (in px) (float32, [3,])

    @staticmethod
    def type() -> str:
        """Returns a string-identifier of the camera model"""
        return "ftheta"

    @property
    def bw_poly(self) -> np.ndarray:
        """Alias for the pixeldistances-to-angles polynomial"""
        return self.pixeldist_to_angle_poly

    @property
    def fw_poly(self) -> np.ndarray:
        """Alias for the angles-to-pixeldistances polynomial"""
        return self.angle_to_pixeldist_poly

    POLYNOMIAL_DEGREE = 6

    def __post_init__(self):
        # Sanity checks
        super().__post_init__()
        assert self.principal_point.shape == (2,)
        assert self.principal_point.dtype == np.dtype("float32")
        assert self.principal_point[0] >= 0.0 and self.principal_point[1] >= 0.0

        assert self.pixeldist_to_angle_poly.ndim == 1
        assert len(self.pixeldist_to_angle_poly) <= self.POLYNOMIAL_DEGREE
        assert self.pixeldist_to_angle_poly.dtype == np.dtype("float32")

        assert self.angle_to_pixeldist_poly.ndim == 1
        assert len(self.angle_to_pixeldist_poly) <= self.POLYNOMIAL_DEGREE
        assert self.angle_to_pixeldist_poly.dtype == np.dtype("float32")

        # pad polynomials to full size
        self.pixeldist_to_angle_poly = np.pad(
            self.pixeldist_to_angle_poly,
            (0, self.POLYNOMIAL_DEGREE - len(self.pixeldist_to_angle_poly)),
            mode="constant",
            constant_values=0.0,
        )
        self.angle_to_pixeldist_poly = np.pad(
            self.angle_to_pixeldist_poly,
            (0, self.POLYNOMIAL_DEGREE - len(self.angle_to_pixeldist_poly)),
            mode="constant",
            constant_values=0.0,
        )

        assert self.max_angle > 0.0

        assert self.linear_cde.shape == (3,)
        assert self.linear_cde.dtype == np.dtype("float32")

    def transform(
        self,
        image_domain_scale: Union[float, Tuple[float, float]],
        image_domain_offset: Tuple[float, float] = (0.0, 0.0),
        new_resolution: Optional[Tuple[int, int]] = None,
    ) -> FThetaCameraModelParameters:
        """
        Applies a transformation to FTheta camera model parameter

        Args:
            image_domain_scale: an isotropic (if float) or anisotropic (if tuple of floats) scaling of the
                                full image domain to a scaled image domain (e.g., to account for up-/downsampling).
                                Resulting scaled image resolution needs to be integer if no explicit 'new_resolution' is provided.
            image_domain_offset: an offset of the _scaled_ image domain (e.g., to account for cropping).
            new_resolution: an optional new resolution to set (if None, the full scaled resolution is used).

        Returns:
            a transformed version of the camera model parameters
        """

        # Get scale factors for each image domain dimension
        image_domain_scale_factors: np.ndarray
        if isinstance(image_domain_scale, tuple):
            image_domain_scale_factors = np.array(image_domain_scale, dtype=np.float32)
        else:
            image_domain_scale_factors = np.array([image_domain_scale, image_domain_scale], dtype=np.float32)

        # Use new resolution if provided
        resolution: np.ndarray
        if new_resolution is not None:
            resolution = np.array(new_resolution, dtype=np.uint64)

        # Otherwise make sure the scaled resolution is integer
        else:
            resolution = self.resolution * image_domain_scale_factors
            assert all([r.is_integer() for r in resolution]), "Resolution must be integer after scaling"

        # Scale / offset principal point location by transforming it in the scaled image (make sure to account for 0.5px offset
        # of the image domain, as the stored parameters are represented with (0,0) at the center of the first pixel)
        principal_point = (
            (self.principal_point + 0.5) * image_domain_scale_factors
            - 0.5
            - np.array(image_domain_offset, dtype=np.float32)
        )

        # Scale bw polynomial by substituting the input pixel domain transformation with the *v-scale*
        # (backwards polynomial is a pixel-distance to angle map, so the domain needs to be scaled).
        # Potentially anisotropic scaling is handled by the linear term.
        scaled_pixel_map = np.polynomial.Polynomial([0.0, 1.0 / image_domain_scale_factors[1]])
        pixeldist_to_angle_poly = np.polynomial.Polynomial(self.pixeldist_to_angle_poly)(scaled_pixel_map).coef.astype(
            np.float32
        )

        # Scale fw polynomial by simple scaling of the result, i.e., linear scaling of the polynomial coefficients
        angle_to_pixeldist_poly = self.angle_to_pixeldist_poly * image_domain_scale_factors[1]

        # Incorporate anisotropic ratio of u/v-scales into the linear term (as the polynomial is unconditionally scaled with the v-scale,
        # and we need to maintain the structure of the linear term [c,d;e,1])
        scale_ratio = image_domain_scale_factors[0] / image_domain_scale_factors[1]
        linear_cde = np.array(
            [self.linear_cde[0] * scale_ratio, self.linear_cde[1] * scale_ratio, self.linear_cde[2]], dtype=np.float32
        )

        # Note: as the FOV can't be effectively increased by scaling / cropping operations, the max-angle is currently not updated and still represents
        # an upper-bound - consider re-computing a tighter upper bound in the future?

        return dataclasses.replace(
            self,
            resolution=resolution.astype(np.uint64),
            principal_point=principal_point,
            pixeldist_to_angle_poly=pixeldist_to_angle_poly,
            angle_to_pixeldist_poly=angle_to_pixeldist_poly,
            linear_cde=linear_cde,
        )


if sys.version_info <= (3, 9):
    # Older python versions have issues with type-hints for nested types in
    # combination with typing.get_type_hints() (used by, e.g., 'dataclasses_json')
    # - alias these globally as a workaround
    PolynomialType = FThetaCameraModelParameters.PolynomialType


@dataclass
class OpenCVPinholeCameraModelParameters(CameraModelParameters, dataclasses_json.DataClassJsonMixin):
    """Represents Pinhole-specific (OpenCV-like) camera model parameters"""

    principal_point: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: U and v coordinate of the principal point, following the :ref:`image coordinate conventions <image_coordinate_conventions>` (float32, [2,])
    focal_length: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: Focal lengths in u and v direction, resp., mapping (distorted) normalized camera coordinates to image coordinates relative to the principal point (float32, [2,])
    radial_coeffs: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: Radial distortion coefficients ``[k1,k2,k3,k4,k5,k6]`` parameterizing the rational radial distortion factor :math:`\frac{1 + k_1r^2 + k_2r^4 + k_3r^6}{1 + k_4r^2 + k_5r^4 + k_6r^6}` for squared norms :math:`r^2` of normalized camera coordinates (float32, [6,])
    tangential_coeffs: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: Tangential distortion coefficients ``[p1,p2]`` parameterizing the tangential distortion components :math:`\begin{bmatrix} 2p_1x'y' + p_2 \left(r^2 + 2{x'}^2 \right) \\ p_1 \left(r^2 + 2{y'}^2 \right) + 2p_2x'y' \end{bmatrix}` for normalized camera coordinates :math:`\begin{bmatrix} x' \\ y' \end{bmatrix}` (float32, [2,])
    thin_prism_coeffs: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: Thins prism distortion coefficients ``[s1,s2,s3,s4]`` parameterizing the thin prism distortion components :math:`\begin{bmatrix} s_1r^2 + s_2r^4 \\ s_3r^2 + s_4r^4 \end{bmatrix}` for squared norms :math:`r^2` of normalized camera coordinates (float32, [4,]

    @staticmethod
    def type() -> str:
        """Returns a string-identifier of the camera model"""
        return "opencv-pinhole"

    def __post_init__(self):
        # Sanity checks
        super().__post_init__()
        assert self.principal_point.shape == (2,)
        assert self.principal_point.dtype == np.dtype("float32")
        assert self.principal_point[0] > 0.0 and self.principal_point[1] > 0.0

        assert self.focal_length.shape == (2,)
        assert self.focal_length.dtype == np.dtype("float32")
        assert self.focal_length[0] > 0.0 and self.focal_length[1] > 0.0

        assert self.radial_coeffs.shape == (6,)
        assert self.radial_coeffs.dtype == np.dtype("float32")

        assert self.tangential_coeffs.shape == (2,)
        assert self.tangential_coeffs.dtype == np.dtype("float32")

        assert self.thin_prism_coeffs.shape == (4,)
        assert self.thin_prism_coeffs.dtype == np.dtype("float32")

    def transform(
        self,
        image_domain_scale: Union[float, Tuple[float, float]],
        image_domain_offset: Tuple[float, float] = (0.0, 0.0),
        new_resolution: Optional[Tuple[int, int]] = None,
    ) -> OpenCVPinholeCameraModelParameters:
        """
        Applies a transformation to OpenCV pinhole camera model parameter

        Args:
            image_domain_scale: an isotropic (if float) or anisotropic (if tuple of floats) scaling of the
                                full image domain to a scaled image domain (e.g., to account for up-/downsampling).
                                Resulting scaled image resolution needs to be integer if no explicit 'new_resolution' is provided.
            image_domain_offset: an offset of the _scaled_ image domain (e.g., to account for cropping).
            new_resolution: an optional new resolution to set (if None, the full scaled resolution is used).

        Returns:
            a transformed version of the camera model parameters
        """

        # Get scale factors for each image domain dimension
        image_domain_scale_factors: np.ndarray
        if isinstance(image_domain_scale, tuple):
            image_domain_scale_factors = np.array(image_domain_scale, dtype=np.float32)
        else:
            image_domain_scale_factors = np.array([image_domain_scale, image_domain_scale], dtype=np.float32)

        # Use new resolution if provided
        resolution: np.ndarray
        if new_resolution is not None:
            resolution = np.array(new_resolution, dtype=np.uint64)

        # Otherwise make sure the scaled resolution is integer
        else:
            resolution = self.resolution * image_domain_scale_factors
            assert all([r.is_integer() for r in resolution]), "Resolution must be integer after scaling"

        return dataclasses.replace(
            self,
            resolution=resolution.astype(np.uint64),
            principal_point=self.principal_point * image_domain_scale_factors
            - np.array(image_domain_offset, dtype=np.float32),
            focal_length=self.focal_length * image_domain_scale_factors,
        )


@dataclass
class OpenCVFisheyeCameraModelParameters(CameraModelParameters, dataclasses_json.DataClassJsonMixin):
    """Represents Fisheye-specific (OpenCV-like) camera model parameters"""

    principal_point: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: U and v coordinate of the principal point, following the :ref:`image coordinate conventions <image_coordinate_conventions>` (float32, [2,])
    focal_length: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: Focal lengths in u and v direction, resp., mapping (distorted) normalized camera coordinates to image coordinates relative to the principal point (float32, [2,])
    radial_coeffs: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: Radial distortion coefficients `radial_coeffs` represent OpenCV-like ``[k1,k2,k3,k4]`` coefficients to parameterize the
    #  fisheye distortion polynomial as :math:`\theta(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8)`
    #  for extrinsic camera ray angles :math:`\theta` with the principal direction (float32, [4,])
    max_angle: float = 0.0  #: Maximal extrinsic ray angle [rad] with the principal direction (float32)

    @staticmethod
    def type() -> str:
        """Returns a string-identifier of the camera model"""
        return "opencv-fisheye"

    def __post_init__(self):
        # Sanity checks
        super().__post_init__()
        assert self.principal_point.shape == (2,)
        assert self.principal_point.dtype == np.dtype("float32")
        assert self.principal_point[0] > 0.0 and self.principal_point[1] > 0.0

        assert self.focal_length.shape == (2,)
        assert self.focal_length.dtype == np.dtype("float32")
        assert self.focal_length[0] > 0.0 and self.focal_length[1] > 0.0

        assert self.radial_coeffs.shape == (4,)
        assert self.radial_coeffs.dtype == np.dtype("float32")

        assert self.max_angle > 0.0

    def transform(
        self,
        image_domain_scale: Union[float, Tuple[float, float]],
        image_domain_offset: Tuple[float, float] = (0.0, 0.0),
        new_resolution: Optional[Tuple[int, int]] = None,
    ) -> OpenCVFisheyeCameraModelParameters:
        """
        Applies a transformation to OpenCV fisheye camera model parameter

        Args:
            image_domain_scale: an isotropic (if float) or anisotropic (if tuple of floats) scaling of the
                                full image domain to a scaled image domain (e.g., to account for up-/downsampling).
                                Resulting scaled image resolution needs to be integer if no explicit 'new_resolution' is provided.
            image_domain_offset: an offset of the _scaled_ image domain (e.g., to account for cropping).
            new_resolution: an optional new resolution to set (if None, the full scaled resolution is used).

        Returns:
            a transformed version of the camera model parameters
        """

        # Get scale factors for each image domain dimension
        image_domain_scale_factors: np.ndarray
        if isinstance(image_domain_scale, tuple):
            image_domain_scale_factors = np.array(image_domain_scale, dtype=np.float32)
        else:
            image_domain_scale_factors = np.array([image_domain_scale, image_domain_scale], dtype=np.float32)

        # Use new resolution if provided
        resolution: np.ndarray
        if new_resolution is not None:
            resolution = np.array(new_resolution, dtype=np.uint64)

        # Otherwise make sure the scaled resolution is integer
        else:
            resolution = self.resolution * image_domain_scale_factors
            assert all([r.is_integer() for r in resolution]), "Resolution must be integer after scaling"

        return dataclasses.replace(
            self,
            resolution=resolution.astype(np.uint64),
            principal_point=self.principal_point * image_domain_scale_factors
            - np.array(image_domain_offset, dtype=np.float32),
            focal_length=self.focal_length * image_domain_scale_factors,
        )


# Represents the collection of all concrete camera model parameter type
ConcreteCameraModelParametersUnion = Union[
    FThetaCameraModelParameters, OpenCVPinholeCameraModelParameters, OpenCVFisheyeCameraModelParameters
]


@dataclass
class Poses:
    """Represents a collection of timestamped poses (rig-to-local-world transformation)"""

    T_rig_world_base: np.ndarray  #: Base rig-to-global-world SE3 transformation (float64, [4,4])
    T_rig_worlds: np.ndarray  #: All rig-to-local-world SE3 transformations of the trajectory (float64, [N,4,4])
    T_rig_world_timestamps_us: np.ndarray  #: All rig-to-local-world transformation timestamps of the trajectory (uint64, [N,])

    def __post_init__(self):
        # Sanity checks
        assert self.T_rig_world_base.shape == (4, 4)
        assert self.T_rig_world_base.dtype == np.dtype("float64")

        assert self.T_rig_worlds.shape[1:] == (4, 4)
        assert self.T_rig_worlds.dtype == np.dtype("float64")

        assert self.T_rig_world_timestamps_us.ndim == 1
        assert self.T_rig_world_timestamps_us.dtype == np.dtype("uint64")

        assert self.T_rig_worlds.shape[0] == self.T_rig_world_timestamps_us.shape[0]


@dataclass
class BBox3(dataclasses_json.DataClassJsonMixin):
    """Parameters of a 3D bounding-box"""

    centroid: Tuple[
        float, float, float
    ]  #: Coordinates [meters] of the bounding-box's centroid in the frame of reference
    dim: Tuple[float, float, float]  #: Extents [meters] of the local bounding-box dimensions in its local frame
    rot: Tuple[
        float, float, float
    ]  #: 'XYZ' Euler rotation angles [radians] orienting the local bounding-box frame to the frame of reference

    def to_array(self) -> np.ndarray:
        """Convert to convenience single-array representation"""
        return np.array(self.centroid + self.dim + self.rot, dtype=np.float32)

    @classmethod
    def from_array(cls, array: np.ndarray) -> BBox3:
        """Convert from convenience single-array representation"""
        return BBox3(
            centroid=(float(array[0]), float(array[1]), float(array[2])),
            dim=(float(array[3]), float(array[4]), float(array[5])),
            rot=(float(array[6]), float(array[7]), float(array[8])),
        )

    def __post_init__(self):
        # Sanity checks
        assert isinstance(self.centroid, tuple)
        assert all(isinstance(i, float) for i in self.centroid)
        assert isinstance(self.dim, tuple)
        assert all(isinstance(i, float) for i in self.dim)
        assert isinstance(self.rot, tuple)
        assert all(isinstance(i, float) for i in self.rot)


@unique
class LabelSource(IntEnum):
    """Enumerates different sources for labels (auto, manual, GT, synthetic etc.)"""

    AUTOLABEL = auto()  #: Label originates from an autolabeling pipeline
    EXTERNAL = auto()  #: Label originates from an unspecified external source, e.g., from third-party processes
    GT_SYNTHETIC = auto()  #: Label originates from a synthetic data simulation and is considered ground-truth
    GT_ANNOTATION = auto()  #: Label originates from manual annotation and is considered ground-truth


@dataclass
class FrameLabel3(dataclasses_json.DataClassJsonMixin):
    """Description of a 3D frame-associated label"""

    label_id: str  #: Identifier of the current frame label (unique among all labels)
    track_id: str  #: Unique identifier of the object's track this label is associated with
    label_class: str  #: String-representation of the class associated with this label
    bbox3: BBox3  #: Bounding-box coordinates of the object relative to the frame's coordinate system
    global_speed: float  #: Instantaneous global speed [m/s] of the object
    timestamp_us: int  #: The timestamp associated with the centroid of the label (possibly an accurate in-spin time)
    confidence: Optional[float]  #: If available, the confidence score of the label [0..1]
    source: LabelSource = util.enum_field(LabelSource)  #: The source for the current label

    def __post_init__(self):
        # Sanity checks
        assert isinstance(self.label_id, str)
        assert isinstance(self.track_id, str)
        assert isinstance(self.label_class, str)
        assert isinstance(self.bbox3, BBox3)
        assert isinstance(self.global_speed, float)
        assert isinstance(self.timestamp_us, int)
        if sys.version_info >= (3, 10):
            assert isinstance(self.confidence, Optional[float])
        else:
            assert isinstance(self.confidence, (type(None), float))
        assert isinstance(self.source, LabelSource)


@dataclass
class TrackLabel(dataclasses_json.DataClassJsonMixin):
    """Description of an individual object-specific track"""

    sensors: Dict[
        str, List[int]
    ]  #: Represents all frame-timestamps (map values) of the object's observations in different sensors (map keys)


@dataclass
class Tracks(dataclasses_json.DataClassJsonMixin):
    """Represents a collection of tracks"""

    track_labels: Dict[
        str, TrackLabel
    ]  #: Represents individual object tracks (map values) referenced by `track_id`'s (map keys, same as in `FrameLabel3`)


@unique
class DynamicFlagState(IntEnum):
    """Enumerates potential per-point flag values related to 'dynamic_flag' property"""

    NOT_AVAILABLE = -1  #: No dynamic flag state is available for this point
    STATIC = 0  #: Point is classified to be static
    DYNAMIC = 1  #: Point is classified to be dynamic


@unique
class FrameTimepoint(IntEnum):
    """Enumerates special timepoints within a frame (values used to index into buffers)"""

    START = 0  #: Requested timepoint is referencing the start time of the frame
    END = 1  #: Requested timepoint is referencing the end time of the frame


class EncodedImageData:
    """Represents encoded image data of a specific format in memory"""

    def __init__(self, encoded_image_data: bytes, encoded_image_format: str):
        self._encoded_image_data = encoded_image_data
        self._encoded_image_format = encoded_image_format

    def get_encoded_image_data(self) -> bytes:
        """Returns encoded image data"""
        return self._encoded_image_data

    def get_encoded_image_format(self) -> str:
        """Returns encoded image format"""
        return self._encoded_image_format

    @lru_cache(maxsize=1)
    def get_decoded_image(self) -> PILImage.Image:
        """Returns decoded image from image data"""
        return PILImage.open(io.BytesIO(self.get_encoded_image_data()), formats=[self.get_encoded_image_format()])


class EncodedImageHandle(Protocol):
    """Protocol type to reference encoded image data (e.g., file-based, container-based, memory-based)"""

    def get_data(self) -> EncodedImageData:
        ...
