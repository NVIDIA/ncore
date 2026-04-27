# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
import io
import sys

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import IntEnum, auto, unique
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

import dataclasses_json
import numpy as np
import PIL.Image as PILImage


if TYPE_CHECKING:
    import numpy.typing as npt  # type: ignore[import-not-found]

from ncore.impl.common.transformations import PoseGraphInterpolator, transform_bbox, transform_point_cloud
from ncore.impl.data import util


if sys.version_info >= (3, 11):
    # Older python versions have issues with type-hints for nested types in
    # combination with typing.get_type_hints() (used by, e.g., 'dataclasses_json')
    # - alias these globally as a workaround
    from typing import Self


## JSON-like structures

JsonLike = Union[
    Dict[str, "JsonLike"],
    List["JsonLike"],
    str,
    int,
    float,
    bool,
    None,
    # special-cases shouldn't be needed, but required to make mypy happy
    List[int],
    List[str],
]


## Data classes representing stored data types
@unique
class ShutterType(IntEnum):
    """Enumerates different possible camera imager shutter types"""

    ROLLING_TOP_TO_BOTTOM = auto()  #: Rolling shutter from top to bottom of the imager
    ROLLING_LEFT_TO_RIGHT = auto()  #: Rolling shutter from left to right of the imager
    ROLLING_BOTTOM_TO_TOP = auto()  #: Rolling shutter from bottom to top of the imager
    ROLLING_RIGHT_TO_LEFT = auto()  #: Rolling shutter from right to left of the imager
    GLOBAL = auto()  #: Instantaneous global shutter (no rolling shutter)


@unique
class ReferencePolynomial(IntEnum):
    """Enumerates different possible reference polynomial types"""

    FORWARD = (
        auto()
    )  #: The forward polynomial is the reference polynomial, the backward polynomial is it's (approximate) inverse
    BACKWARD = (
        auto()
    )  #: The backward polynomial is the reference polynomial, the forward polynomial is it's (approximate) inverse


@dataclass
class BivariateWindshieldModelParameters(dataclasses_json.DataClassJsonMixin):
    """Represents parameters required to create a windshield external distortion model"""

    reference_poly: ReferencePolynomial = util.enum_field(ReferencePolynomial)  #: Reference polynomial of the model

    # Forward correction coefficients (project to sensor)
    horizontal_poly: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: Polynomial coefficients used for forward projection on the horizontal component of a ray via it's projected angle phi=asin(x/norm(x,y)). The polynomial is of order N in both phi and theta with the form P(phi,N)*P(theta,0) + P(phi, N-1)*P(theta,1) ... + P(phi, N-N)*P(theta,N), where P(i, N) is a polynomial over "i" of degree N (float32, [(N + 1) * (N + 2) / 2,])
    vertical_poly: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: Polynomial coefficients used for forward projection on the vertical component of a ray via it's projected angle theta=asin(y/norm(x,y)). The polynomial is of order M in both phi and theta with the form P(phi,M)*P(theta,0) + P(phi, M-1)*P(theta,1) ... + P(phi, M-M)*P(theta,M), where P(i, M) is a polynomial over "i" of degree M (float32, [(M + 1) * (M + 2) / 2,])

    # Backward correction coefficient (project to world)
    horizontal_poly_inverse: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: Polynomial coefficients used to evaluate the inverse distortion in backprojection of the horizontal component of a ray via it's projected angle phi=asin(x/norm(x,y)). The polynomial is of order N in both phi and theta with the form P(phi,N)*P(theta,0) + P(phi, N-1)*P(theta,1) ... + P(phi, N-N)*P(theta,N), where P(i, N) is a polynomial over "i" of degree N (float32, [(N + 1) * (N + 2) / 2,])

    vertical_poly_inverse: np.ndarray = util.numpy_array_field(
        np.float32
    )  #: Polynomial coefficients used to evaluate the inverse distortion in backprojection of the vertical component of a ray via it's projected angle theta=asin(y/norm(x,y)). The polynomial is of order M in both phi and theta with the form P(phi,M)*P(theta,0) + P(phi, M-1)*P(theta,1) ... + P(phi, M-M)*P(theta,M), where P(i, M) is a polynomial over "i" of degree M (float32, [(M + 1) * (M + 2) / 2,])

    @staticmethod
    def type() -> str:
        """Returns a string-identifier of the external distortion model"""
        return "bivariate-windshield"

    def __post_init__(self):
        # Sanity checks
        assert isinstance(self.reference_poly, ReferencePolynomial)

        assert self.horizontal_poly.ndim == 1
        assert self.horizontal_poly.dtype == np.dtype("float32")

        assert self.vertical_poly.ndim == 1
        assert self.horizontal_poly.dtype == np.dtype("float32")

        assert self.horizontal_poly_inverse.ndim == 1
        assert self.horizontal_poly.dtype == np.dtype("float32")

        assert self.vertical_poly_inverse.ndim == 1
        assert self.horizontal_poly.dtype == np.dtype("float32")


# Represents the collection of all concrete external distortion types
ConcreteExternalDistortionParametersUnion = Union[BivariateWindshieldModelParameters]

# Self type-var for camera model parameters consistent with PEP 673 but compatible with Python < 3.11
CameraModelParametersSelf = TypeVar("CameraModelParametersSelf", bound="CameraModelParameters")


@dataclass
class CameraModelParameters(ABC):
    """Represents parameters common to all camera models"""

    resolution: np.ndarray = util.numpy_array_field(
        np.uint64
    )  #: Width and height of the image in pixels (uint64, [2,])
    shutter_type: ShutterType = util.enum_field(ShutterType)  #: Shutter type of the camera's imaging sensor

    external_distortion_parameters: Optional[ConcreteExternalDistortionParametersUnion] = (
        None  #: Optional external distortion source associated to the camera (e.g. windshield). If a source exits, rays will be distorted prior to reaching the camera and it's associated lens distortion if applicable
    )

    @abstractmethod
    def transform(
        self: CameraModelParametersSelf,
        image_domain_scale: Union[float, Tuple[float, float]],
        image_domain_offset: Tuple[float, float] = (0.0, 0.0),
        new_resolution: Optional[Tuple[int, int]] = None,
    ) -> CameraModelParametersSelf:
        """
        Applies a transformation to camera model parameter

        Args:
            image_domain_scale: an isotropic (if float) or anisotropic (if tuple of floats) scaling of the
                                full image domain to a scaled image domain (e.g., to account for up-/downsampling).
                                Resulting scaled image resolution needs to be integer if no explicit 'new_resolution' is provided.
            image_domain_offset: an offset of the _scaled_ image domain (e.g., to account for cropping).
            new_resolution: an optional new resolution to set (if None, the full scaled resolution is used).

        Returns:
            a transformed version of the concrete camera model parameters
        """

    def __post_init__(self):
        # Sanity checks
        assert self.resolution.shape == (2,)
        assert self.resolution.dtype == np.dtype("uint64")
        assert self.resolution[0] > 0 and self.resolution[1] > 0

        if not isinstance(self.shutter_type, ShutterType):
            self.shutter_type = ShutterType(self.shutter_type)
        assert self.shutter_type in ShutterType.__members__.values()

        assert isinstance(self.external_distortion_parameters, (type(None), ConcreteExternalDistortionParametersUnion))


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

        if not isinstance(self.reference_poly, FThetaCameraModelParameters.PolynomialType):
            self.reference_poly = FThetaCameraModelParameters.PolynomialType(self.reference_poly)
        assert self.reference_poly in FThetaCameraModelParameters.PolynomialType.__members__.values()

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

        # some datasets might store _invalid_ linear terms (all zero) - workaround by setting these to default linear term
        if np.allclose(self.linear_cde, 0.0):
            self.linear_cde = np.array([1.0, 0.0, 0.0], dtype=np.float32)

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
            a transformed version of the FTheta camera model parameters
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
            a transformed version of the OpenCV pinhole camera model parameters
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
            a transformed version of the OpenCV fisheye camera model parameters
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


def encode_camera_model_parameters(camera_model_parameters: ConcreteCameraModelParametersUnion) -> Dict:
    """Encodes camera intrinsic model parameters to serializable model-typed dictionary"""

    encoded = {
        "camera_model_type": camera_model_parameters.type(),
        "camera_model_parameters": camera_model_parameters.to_dict(),
    }

    # Store type of external distortion, if available
    if camera_model_parameters.external_distortion_parameters:
        encoded["external_distortion_type"] = camera_model_parameters.external_distortion_parameters.type()

    return encoded


def decode_camera_model_parameters(encoded_parameters: Mapping) -> ConcreteCameraModelParametersUnion:
    """Decodes model-typed dictionary parameters specific to the camera's intrinsic model"""

    camera_model_type = encoded_parameters["camera_model_type"]

    # Copy as we might modify the dictionary in place
    camera_model_parameters = encoded_parameters["camera_model_parameters"].copy()

    # Hook up typed external distortion type, if present
    external_distortion_type: Optional[str] = encoded_parameters.get("external_distortion_type")
    if external_distortion_type is not None:
        if external_distortion_type == "bivariate-windshield":
            camera_model_parameters["external_distortion_parameters"] = BivariateWindshieldModelParameters.from_dict(
                camera_model_parameters["external_distortion_parameters"]
            )
        else:
            raise ValueError(f"Unknown external distortion type: {external_distortion_type}")

    # Return typed camera model parameters
    if camera_model_type == "ftheta":
        return FThetaCameraModelParameters.from_dict(camera_model_parameters)
    elif camera_model_type in [
        "opencv-pinhole",
        # keep 'pinhole' for backwards-compatibility with existing data
        "pinhole",
    ]:
        return OpenCVPinholeCameraModelParameters.from_dict(camera_model_parameters)
    elif camera_model_type == "opencv-fisheye":
        return OpenCVFisheyeCameraModelParameters.from_dict(camera_model_parameters)

    raise ValueError(f"Unknown camera model type: {camera_model_type}")


@dataclass()
class BaseLidarModelParameters:
    """Represents parameters common to all lidar models"""

    pass


@dataclass()
class BaseSpinningLidarModelParameters(BaseLidarModelParameters):
    """Represents parameters common to all spinning lidar models"""

    spinning_frequency_hz: float  # spinning frequency / frames per second [Hz]

    spinning_direction: Literal[
        "cw", "ccw"
    ]  # direction of spinning, either clockwise (cw) or counter-clockwise (ccw) [around z axis]

    def __post_init__(self):
        # Sanity checks
        assert self.spinning_frequency_hz > 0.0
        assert self.spinning_direction in ["cw", "ccw"]


@dataclass()
class BaseStructuredSpinningLidarModelParameters(BaseSpinningLidarModelParameters):
    """Represents parameters for a structured spinning lidar model.

    A structured lidar model consists of a fixed number of rows x columns point measurements per frame
    """

    n_rows: int  # number of rows
    n_columns: int  # number of columns

    def __post_init__(self):
        # Sanity checks
        assert self.n_rows > 0
        assert self.n_columns > 0


@dataclass()
class RowOffsetStructuredSpinningLidarModelParameters(
    BaseStructuredSpinningLidarModelParameters, dataclasses_json.DataClassJsonMixin
):
    """Represents parameters for a structured spinning lidar model that is using a per-row azimuth-offset (compatible with, e.g., Hesai P128 sensors)"""

    # elevation angles
    row_elevations_rad: np.ndarray = util.numpy_array_field(
        np.float32
    )  # elevation angle of each row, constant for each column [clockwise around y axis, relative to x axis] [(Nrows,) radians]

    # azimuth angles
    column_azimuths_rad: np.ndarray = util.numpy_array_field(
        np.float32
    )  # azimuth angle of each column, starting at first element of the spin [clockwise / counter-clockwise around z axis depending on sensors spin direction, relative to x axis] [(Ncolumns,) radians]
    row_azimuth_offsets_rad: np.ndarray = util.numpy_array_field(
        np.float32
    )  # azimuth angle offsets for each row (optional, can be zero if no row offsets) [around z axis, relative to x axis] [(Nrows,) radians]

    def __post_init__(self):
        # Sanity checks

        assert self.row_elevations_rad.dtype == np.float32
        assert self.row_elevations_rad.shape == (self.n_rows,)
        assert self.row_azimuth_offsets_rad.dtype == np.float32
        assert self.row_azimuth_offsets_rad.shape == (self.n_rows,)
        assert self.column_azimuths_rad.dtype == np.float32
        assert self.column_azimuths_rad.shape == (self.n_columns,)

        # Check elevation angles are sorted consistently
        relative_row_elevations_rad = util.relative_angle(self.row_elevations_rad[0], self.row_elevations_rad, "cw")
        assert np.all(np.diff(relative_row_elevations_rad.relative_angle_rad) > 0), (
            "Row elevation angles must be sorted in descending order (cw)"
        )
        assert np.all(~relative_row_elevations_rad.wrap_around_flag), (
            "Row elevation angles must not wrap around the start element"
        )

        # Check order of column azimuth angles is consistent with spinning direction
        relative_column_azimuths_rad = util.relative_angle(
            self.column_azimuths_rad[0], self.column_azimuths_rad, self.spinning_direction
        )
        assert np.all(np.diff(relative_column_azimuths_rad.relative_angle_rad) > 0), (
            "Column azimuth angles must be sorted in the spinning direction so the diff between relative angles of consecutive columns should always be positive"
        )
        assert np.all(~relative_row_elevations_rad.wrap_around_flag), (
            "Column azimuth angles (without offsets) must not wrap around the start element"
        )

    @staticmethod
    def type() -> str:
        """Returns a string-identifier of the lidar model"""
        return "row-offset-spinning"

    def get_vertical_fov(self, dtype: "npt.DTypeLike" = np.float32) -> util.FOV:
        """Returns the vertical field-of-view of the lidar model (starting at first element) in the requested dtype precision"""

        start_rad = self.row_elevations_rad[0].astype(dtype).item()
        span_rad = util.relative_angle(
            start_rad, self.row_elevations_rad.astype(dtype)[-1], "cw"
        ).relative_angle_rad.item()

        return util.FOV(start_rad=start_rad, span_rad=span_rad, direction="cw")

    def get_horizontal_fov(self, dtype: "npt.DTypeLike" = np.float32) -> util.FOV:
        """Returns the horizontal field-of-view of the lidar model (starting at first element) in the requested dtype precision"""

        # Reconstruct first and last (wrapped) element azimuths once to obtain FoV bounds
        azimuths_rad = (
            self.column_azimuths_rad.astype(dtype)[None, [0, self.n_columns - 1]]
            + self.row_azimuth_offsets_rad.astype(dtype)[:, None]
        )

        # Determine extremum in first element
        if self.spinning_direction == "ccw":
            start_rad = azimuths_rad[:, 0].min().item()
        else:
            start_rad = azimuths_rad[:, 0].max().item()

        # Check if the azimuth angles of last element wrap around over the start element
        span = util.relative_angle(start_rad, azimuths_rad[:, -1], self.spinning_direction)
        if np.any(span.wrap_around_flag):
            span_rad = 2 * np.pi
        else:
            span_rad = span.relative_angle_rad.max().item()

        return util.FOV(start_rad=start_rad, span_rad=span_rad, direction=self.spinning_direction)


# Represents the collection of all concrete lidar model parameter type
ConcreteLidarModelParametersUnion = Union[RowOffsetStructuredSpinningLidarModelParameters]


def encode_lidar_model_parameters(lidar_model_parameters: ConcreteLidarModelParametersUnion) -> Dict:
    """Encodes lidar intrinsic model parameters to serializable model-typed dictionary"""

    encoded = {
        "lidar_model_type": lidar_model_parameters.type(),
        "lidar_model_parameters": lidar_model_parameters.to_dict(),
    }

    return encoded


def decode_lidar_model_parameters(encoded_parameters: Mapping) -> ConcreteLidarModelParametersUnion:
    """Decodes model-typed dictionary parameters specific to the lidars's intrinsic model"""

    lidar_model_type = encoded_parameters["lidar_model_type"]

    # Return typed lidar model parameters
    if lidar_model_type == RowOffsetStructuredSpinningLidarModelParameters.type():
        return RowOffsetStructuredSpinningLidarModelParameters.from_dict(encoded_parameters["lidar_model_parameters"])

    raise ValueError(f"Unknown lidar model type: {lidar_model_type}")


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
class CuboidTrackObservation(dataclasses_json.DataClassJsonMixin):
    """Individual cuboid track observation relative to a reference frame"""

    track_id: str  #: Unique identifier of the object's track this observation is associated with
    class_id: str  #: String-representation of the labeled class of the object

    timestamp_us: (
        int  #: The timestamp associated with the centroid of the observation (possibly an accurate in-frame time)
    )

    reference_frame_id: str  #: String-identifier of the reference frame (e.g., sensor name)
    reference_frame_timestamp_us: int  #: The timestamp of the reference frame

    bbox3: BBox3  #: Bounding-box coordinates of the object relative to the reference frame's coordinate system

    source: LabelSource = util.enum_field(LabelSource)  #: The source for the current label
    source_version: Optional[str] = (
        None  #: If provided, the unique version ID of the source for the current label (to distinguish between different versions of the same source)
    )

    def transform(
        self,
        target_frame_id: str,
        target_frame_timestamp_us: int,
        pose_graph: PoseGraphInterpolator,
        anchor_frame_id: str = "world",
    ) -> "Self":
        """Transform the observation's bounding box to a different reference frame.

        Args:
            target_frame_id: ID of the target reference frame
            target_frame_timestamp_us: Timestamp of the target reference frame
            pose_graph: PoseGraphInterpolator to perform the evaluation of transformations
            anchor_frame_id: ID of the common anchor frame for transformations (default: "world")

        Returns:
            A CuboidTrackObservation instance with the transformed bounding box and updated reference frame info
        """

        if (
            self.reference_frame_id == target_frame_id
            and self.reference_frame_timestamp_us == target_frame_timestamp_us
        ):
            # Skip transformation if already in correct target frame
            return self

        # Transform observation from reference frame at observation time to target frame at target time via world
        T_reference_world = pose_graph.evaluate_poses(
            self.reference_frame_id,
            anchor_frame_id,
            np.array(self.reference_frame_timestamp_us, dtype=np.int64),
        )
        T_world_target = pose_graph.evaluate_poses(
            anchor_frame_id,
            target_frame_id,
            np.array(target_frame_timestamp_us, dtype=np.int64),
        )

        T_reference_target = T_world_target @ T_reference_world

        return replace(
            self,
            bbox3=BBox3.from_array(transform_bbox(self.bbox3.to_array(), T_reference_target)),
            reference_frame_id=target_frame_id,
            reference_frame_timestamp_us=target_frame_timestamp_us,
        )

    def __post_init__(self):
        # Sanity checks
        assert isinstance(self.track_id, str)
        assert isinstance(self.class_id, str)
        assert isinstance(self.reference_frame_id, str)
        assert isinstance(self.reference_frame_timestamp_us, int)
        assert isinstance(self.bbox3, BBox3)
        assert isinstance(self.timestamp_us, int)

        if not isinstance(self.source, LabelSource):
            self.source = LabelSource(self.source)
        assert self.source in LabelSource.__members__.values()

        assert isinstance(self.source_version, (type(None), str))


@unique
class FrameTimepoint(IntEnum):
    """Enumerates special timepoints within a frame (values used to index into buffers)"""

    START = 0  #: Requested timepoint is referencing the start time of the frame
    END = 1  #: Requested timepoint is referencing the end time of the frame


@dataclass(**({"slots": True, "frozen": True} if sys.version_info >= (3, 10) else {"frozen": True}))
class PointCloud:
    """Immutable point cloud with lazy attribute loading and rigid-transform support.

    All points share a single reference frame and timestamp (the snapshot timestamp).
    Per-point timestamps, if available, are stored as a regular ``INVARIANT`` attribute.

    The :meth:`transform` method returns a new :class:`PointCloud` with an accumulated
    rigid transform stored in ``_T_raw_reference``.  The :attr:`xyz` property and
    covariant attributes apply this transform lazily on access.
    """

    # -- nested types -----------------------------------------------------------

    class AttributeTransformType(IntEnum):
        """How an attribute behaves under a rigid transformation."""

        INVARIANT = 0  #: Unchanged (rgb, intensity, confidence, label_id, timestamp_us)
        DIRECTION = 1  #: Rotate only: R @ v (normals, flow directions)
        POINT = 2  #: Full rigid: R @ p + t (secondary xyz positions)

    class CoordinateUnit(IntEnum):
        """Physical unit of the point coordinates."""

        UNITLESS = 0  #: Arbitrary / unknown scale (e.g. SfM reconstruction)
        METERS = 1  #: Metric (meters)

    @dataclass(**({"slots": True, "frozen": True} if sys.version_info >= (3, 10) else {"frozen": True}))
    class Attribute:
        """A lazily-loaded point-cloud attribute together with its transform semantics."""

        loader: Callable[[], "npt.NDArray"]
        transform_type: PointCloud.AttributeTransformType

    # -- fields ----------------------------------------------------------------

    _xyz: "npt.NDArray[np.floating]"
    reference_frame_id: str
    reference_frame_timestamp_us: int
    coordinate_unit: PointCloud.CoordinateUnit
    _attributes: Dict[str, PointCloud.Attribute]
    _T_raw_reference: "Optional[npt.NDArray[np.floating]]" = dataclasses.field(default=None)

    # -- properties ------------------------------------------------------------

    @property
    def points_count(self) -> int:
        """Number of points in the cloud."""
        return int(self._xyz.shape[0])

    @property
    def xyz(self) -> "npt.NDArray":
        """Points in the target reference frame (transform applied lazily)."""
        if self._T_raw_reference is None:
            return self._xyz
        return transform_point_cloud(self._xyz, self._T_raw_reference)

    @property
    def attribute_names(self) -> List[str]:
        """Names of all registered attributes."""
        return list(self._attributes.keys())

    # -- methods ---------------------------------------------------------------

    def has_attribute(self, name: str) -> bool:
        """Returns ``True`` if an attribute with the given *name* is registered."""
        return name in self._attributes

    def get_attribute(self, name: str) -> "npt.NDArray":
        """Load and return an attribute, applying the accumulated transform if applicable.

        Raises:
            KeyError: if *name* does not refer to a known attribute.
        """
        if name not in self._attributes:
            raise KeyError(f"Unknown attribute: {name}")

        attr = self._attributes[name]
        raw: "npt.NDArray" = attr.loader()

        if self._T_raw_reference is None or attr.transform_type == self.AttributeTransformType.INVARIANT:
            return raw

        if attr.transform_type == self.AttributeTransformType.DIRECTION:
            R = self._T_raw_reference[:3, :3]
            return (R @ raw.T).T

        # POINT
        return transform_point_cloud(raw, self._T_raw_reference)

    def get_attribute_transform_type(self, name: str) -> PointCloud.AttributeTransformType:
        """Return the :class:`AttributeTransformType` for *name*.

        Raises:
            KeyError: if *name* does not refer to a known attribute.
        """
        if name not in self._attributes:
            raise KeyError(f"Unknown attribute: {name}")
        return self._attributes[name].transform_type

    def transform(
        self,
        target_frame_id: str,
        target_frame_timestamp_us: int,
        pose_graph: PoseGraphInterpolator,
        anchor_frame_id: str = "world",
    ) -> "PointCloud":
        """Transform this point cloud to a different reference frame.

        Signature is aligned with :meth:`CuboidTrackObservation.transform`.

        The transform is **not** applied eagerly -- it is stored and applied lazily
        when :attr:`xyz` or :meth:`get_attribute` are accessed.

        Args:
            target_frame_id: ID of the target reference frame.
            target_frame_timestamp_us: Timestamp of the target reference frame.
            pose_graph: PoseGraphInterpolator to evaluate transformations.
            anchor_frame_id: ID of the common anchor frame (default: ``"world"``).

        Returns:
            A new PointCloud with updated reference frame and an accumulated lazy transform.
        """
        if (
            self.reference_frame_id == target_frame_id
            and self.reference_frame_timestamp_us == target_frame_timestamp_us
        ):
            return self

        # Compute transform from current reference frame to new target frame
        T_currentref_anchor = pose_graph.evaluate_poses(
            self.reference_frame_id,
            anchor_frame_id,
            np.array(self.reference_frame_timestamp_us, dtype=np.uint64),
        )
        T_anchor_target = pose_graph.evaluate_poses(
            anchor_frame_id,
            target_frame_id,
            np.array(target_frame_timestamp_us, dtype=np.uint64),
        )
        T_currentref_target = T_anchor_target @ T_currentref_anchor

        # Compose with existing accumulated transform: raw -> current_ref -> target
        if self._T_raw_reference is not None:
            T_raw_target = T_currentref_target @ self._T_raw_reference
        else:
            T_raw_target = T_currentref_target

        return PointCloud(
            _xyz=self._xyz,
            reference_frame_id=target_frame_id,
            reference_frame_timestamp_us=target_frame_timestamp_us,
            coordinate_unit=self.coordinate_unit,
            _attributes=self._attributes,
            _T_raw_reference=T_raw_target,
        )


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

    def get_data(self) -> EncodedImageData: ...


# ---------------------------------------------------------------------------
#  Label type system
# ---------------------------------------------------------------------------


@unique
class LabelCategory(IntEnum):
    """High-level category of a label types."""

    DEPTH = 0  #: Distance measures (z-axis, ray, relative)
    FLOW = 1  #: Motion displacement fields (optical, scene)
    SEGMENTATION = 2  #: Classification (semantic, instance, panoptic)
    MASK = 3  #: Binary / multi-level masks
    GEOMETRY = 4  #: Geometric vectors (normals, ray directions)
    MATERIAL = 5  #: Material / surface properties (albedo, roughness)
    FEATURE = 6  #: Feature embeddings (DINOv2, CLIP)
    OTHER = 7  #: Catch-all for uncategorised labels
    UNKNOWN = -1  #: Unrecognised / fallback category (reader-only)

    @classmethod
    def resolve(cls, name: str) -> LabelCategory:
        """Return the member matching *name*, or :py:attr:`UNKNOWN` if unrecognised."""
        try:
            return cls.__members__[name]
        except KeyError:
            return cls.UNKNOWN


@unique
class LabelUnit(IntEnum):
    """Physical unit associated with a label's numeric values."""

    METERS = 0  #: Metric (meters)
    PIXELS = 1  #: Pixel displacement
    UNITLESS = 2  #: Dimensionless quantity (e.g. class IDs, masks)
    UNKNOWN = -1  #: Unrecognised / fallback unit

    @classmethod
    def resolve(cls, name: str) -> LabelUnit:
        """Return the member matching *name*, or :py:attr:`UNKNOWN` if unrecognised."""
        try:
            return cls.__members__[name]
        except KeyError:
            return cls.UNKNOWN


@unique
class LabelEncoding(IntEnum):
    """Describes how the raw label data is stored on disk."""

    RAW = 0  #: Stored as a raw numeric array
    IMAGE_ENCODED = 1  #: Stored as an encoded image (e.g. PNG, JPEG)
    UNKNOWN = -1  #: Unrecognised / fallback encoding

    @classmethod
    def resolve(cls, name: str) -> LabelEncoding:
        """Return the member matching *name*, or :py:attr:`UNKNOWN` if unrecognised."""
        try:
            return cls.__members__[name]
        except KeyError:
            return cls.UNKNOWN


@dataclass(**({"slots": True, "frozen": True} if sys.version_info >= (3, 10) else {"frozen": True}))
class LabelType(dataclasses_json.DataClassJsonMixin):
    """Describes the semantic kind of a label: category + qualifier + unit.

    Well-known combinations are exposed as class-level constants (e.g. ``LabelType.DEPTH_Z``).
    Project-specific labels use custom qualifiers with no code changes required.
    """

    category: LabelCategory = util.enum_field(LabelCategory)  #: High-level label family
    qualifier: str = ""  #: Free-form variant identifier (e.g. "z", "optical_forward", "semantic")
    unit: Optional[LabelUnit] = dataclasses.field(
        default=None,
        metadata=dataclasses_json.config(
            encoder=lambda u: u.name if u is not None else None,
            decoder=lambda s: LabelUnit.resolve(s) if s is not None else None,
        ),
    )  #: Physical unit of the label values

    # Well-known constants (assigned after class definition)
    DEPTH_Z: ClassVar[LabelType]
    DEPTH_RAY: ClassVar[LabelType]
    DEPTH_RELATIVE: ClassVar[LabelType]
    FLOW_OPTICAL_FORWARD: ClassVar[LabelType]
    FLOW_OPTICAL_BACKWARD: ClassVar[LabelType]
    FLOW_SCENE_FORWARD: ClassVar[LabelType]
    FLOW_SCENE_BACKWARD: ClassVar[LabelType]
    SEGMENTATION_SEMANTIC: ClassVar[LabelType]
    SEGMENTATION_INSTANCE: ClassVar[LabelType]
    MASK_BACKGROUND: ClassVar[LabelType]
    MASK_DYNAMIC: ClassVar[LabelType]
    GEOMETRY_NORMAL_CAMERA: ClassVar[LabelType]
    GEOMETRY_NORMAL_WORLD: ClassVar[LabelType]
    GEOMETRY_RAY_DIRECTION: ClassVar[LabelType]


# Well-known LabelType constants (set after class definition)
LabelType.DEPTH_Z = LabelType(LabelCategory.DEPTH, "z", LabelUnit.METERS)
LabelType.DEPTH_RAY = LabelType(LabelCategory.DEPTH, "ray", LabelUnit.METERS)
LabelType.DEPTH_RELATIVE = LabelType(LabelCategory.DEPTH, "relative", LabelUnit.UNITLESS)
LabelType.FLOW_OPTICAL_FORWARD = LabelType(LabelCategory.FLOW, "optical_forward", LabelUnit.PIXELS)
LabelType.FLOW_OPTICAL_BACKWARD = LabelType(LabelCategory.FLOW, "optical_backward", LabelUnit.PIXELS)
LabelType.FLOW_SCENE_FORWARD = LabelType(LabelCategory.FLOW, "scene_forward", LabelUnit.METERS)
LabelType.FLOW_SCENE_BACKWARD = LabelType(LabelCategory.FLOW, "scene_backward", LabelUnit.METERS)
LabelType.SEGMENTATION_SEMANTIC = LabelType(LabelCategory.SEGMENTATION, "semantic")
LabelType.SEGMENTATION_INSTANCE = LabelType(LabelCategory.SEGMENTATION, "instance")
LabelType.MASK_BACKGROUND = LabelType(LabelCategory.MASK, "background")
LabelType.MASK_DYNAMIC = LabelType(LabelCategory.MASK, "dynamic")
LabelType.GEOMETRY_NORMAL_CAMERA = LabelType(LabelCategory.GEOMETRY, "normal_camera", LabelUnit.UNITLESS)
LabelType.GEOMETRY_NORMAL_WORLD = LabelType(LabelCategory.GEOMETRY, "normal_world", LabelUnit.UNITLESS)
LabelType.GEOMETRY_RAY_DIRECTION = LabelType(LabelCategory.GEOMETRY, "ray_direction")


@dataclass(**({"slots": True, "frozen": True} if sys.version_info >= (3, 10) else {"frozen": True}))
class QuantizationParams(dataclasses_json.DataClassJsonMixin):
    """Parameters for de-quantizing stored integer data back to physical values.

    The physical value is recovered as ``value = stored * scale + offset``.
    """

    quantized_dtype: np.dtype = util.dtype_field()  #: Numpy dtype of the quantized on-disk representation
    scale: float = 1.0  #: Multiplicative scale factor
    offset: float = 0.0  #: Additive offset


@dataclass(**({"slots": True, "frozen": True} if sys.version_info >= (3, 10) else {"frozen": True}))
class LabelSchema(dataclasses_json.DataClassJsonMixin):
    """Schema describing the dtype, shape, encoding and quantization of a single label layer."""

    dtype: np.dtype = util.dtype_field()  #: Numpy dtype of the label data (after decoding / de-quantization)
    shape_suffix: Tuple[int, ...] = dataclasses.field(
        default=(),
        metadata=dataclasses_json.config(encoder=list, decoder=tuple),
    )  #: Extra dimensions appended to (H, W)
    encoding: LabelEncoding = util.enum_field(LabelEncoding)  #: How the label data is stored on disk
    encoded_format: Optional[str] = (
        None  #: Image format string (e.g. ``"png"``, ``"jpeg"``) when ``encoding == IMAGE_ENCODED``
    )
    quantization: Optional[QuantizationParams] = None  #: Optional quantization parameters


@dataclass(**({"slots": True, "frozen": True} if sys.version_info >= (3, 10) else {"frozen": True}))
class CameraLabelDescriptor(dataclasses_json.DataClassJsonMixin):
    """Compound descriptor bundling the identity and schema of one camera label instance.

    Passed directly to :class:`CameraLabelsComponent.Writer` to define what it stores.
    The :attr:`default_instance_name` property provides a recommended naming convention.
    """

    camera_id: str = ""  #: Associated camera identifier
    label_type: LabelType = dataclasses.field(default_factory=lambda: LabelType(LabelCategory.UNKNOWN, ""))
    label_schema: LabelSchema = dataclasses.field(
        default_factory=lambda: LabelSchema(dtype=np.dtype("float32"), shape_suffix=(), encoding=LabelEncoding.RAW)
    )

    @property
    def default_instance_name(self) -> str:
        """Recommended instance name: ``category.qualifier@camera_id``."""
        cat = self.label_type.category.name.lower()
        return f"{cat}.{self.label_type.qualifier}@{self.camera_id}"
