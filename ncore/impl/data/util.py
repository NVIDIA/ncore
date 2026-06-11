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

import re

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Iterable, List, Literal, Optional, TypeVar, cast

import dataclasses_json
import numpy as np

from numpy.polynomial.polynomial import Polynomial
from upath import UPath


if TYPE_CHECKING:
    import numpy.typing as npt  # type: ignore[import-not-found]

## Constants
INDEX_DIGITS = 6  # the number of integer digits to pad counters in output filenames


## Types
@dataclass
class FOV:
    """Represents a field-of-view with start and span in radians"""

    start_rad: float  #: Start angle of the field-of-view in radians
    span_rad: float  #: Span of the valid field-of-view region in radians in [0, 2π]
    direction: Literal[
        "cw", "ccw"
    ]  #: Direction of the valid field-of-view region, either clockwise or counter-clockwise


## Functions
def padded_index_string(index: int, index_digits=INDEX_DIGITS) -> str:
    """Pads an integer with leading zeros to a fixed number of digits"""
    return str(index).zfill(index_digits)


def closest_index_sorted(sorted_array: np.ndarray, value: int) -> int:
    """Returns the index of the closest value within a *sorted* array relative to a query value.

    Note: we are *not* checking that the input is sorted
    """
    if not len(sorted_array):
        raise ValueError("input array is empty")

    idx = int(np.searchsorted(sorted_array, value, side="left"))

    if idx > 0:
        if idx == len(sorted_array):
            return idx - 1
        if abs(value - sorted_array[idx - 1]) < abs(sorted_array[idx] - value):
            return idx - 1

    return idx


def numpy_array_field(datatype: "npt.DTypeLike", default=None) -> Any:
    """Provides encoder / decoder functionality for numpy arrays into field types compatible with dataclass-JSON"""

    def decoder(*args, **kwargs) -> np.ndarray:
        return np.array(*args, **kwargs).astype(datatype)

    metadata = dataclasses_json.config(encoder=np.ndarray.tolist, decoder=decoder)

    if default is not None:
        return field(default_factory=lambda: default, metadata=metadata)
    else:
        return field(default=None, metadata=metadata)


def enum_field(enum_class, default=None) -> Any:
    """Provides encoder / decoder functionality for enum types into field types compatible with dataclass-JSON"""

    def encoder(variant) -> str:
        """encode enum as name's string representation. This way values in JSON are "human-readable"""
        return variant.name

    def decoder(variant) -> Any:
        """load enum variant from name's string to value map of the enumeration type"""
        return enum_class.__members__[variant]

    return field(default=default, metadata=dataclasses_json.config(encoder=encoder, decoder=decoder))


def dtype_field(default: Optional[np.dtype] = None) -> Any:
    """Provides encoder / decoder functionality for numpy dtype fields into field types compatible with dataclass-JSON.

    Serializes as the dtype's string name (e.g., ``"float32"``, ``"uint8"``).
    """

    return field(
        default=default,
        metadata=dataclasses_json.config(
            encoder=lambda d: str(d),
            decoder=lambda s: np.dtype(s),
        ),
    )


def evaluate_file_pattern(pattern: str, skip_suffixes: Iterable[str] = ()) -> List[str]:
    """Given a file-pattern returns a list of matching and existing files

    Supported patterns (mutually exclusive):
    - integer-ranges: '/some/path/file-[1-3]' will be expanded to [/some/path/file-1, /some/path/file-2, /some/path/file-3]

    """

    pattern_basepath = UPath(pattern).parent
    pattern_name = UPath(pattern).name

    evaluated_name_patterns = []

    # expand integer ranges like '[1-13]'
    if range_match := re.search(r"\[(\d+)-(\d+)\]", pattern_name):
        low = int(range_match.group(1))
        high = int(range_match.group(2))

        for i in range(low, high + 1):
            evaluated_name_patterns.append(pattern_name.replace(f"[{low}-{high}]", str(i) + "-"))
    else:
        evaluated_name_patterns.append(pattern_name)

    matches: set[UPath] = set()
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

    return [str(match) for match in list(matches)]


# A generic type supporting basic artithmetic operations like +, -, *, /, etc. - in particular implemented by float, torch.Tensor, np.ndarray, etc.
# Used here to not depend on torch.Tensor in the public data API
TensorLike = TypeVar("TensorLike", bound=Any)


@dataclass
class RelativeAngleResult(Generic[TensorLike]):
    relative_angle_rad: TensorLike
    wrap_around_flag: TensorLike


def relative_angle(
    ref_angle_rad: float, angle_rad: TensorLike, direction: Literal["cw", "ccw"]
) -> "RelativeAngleResult[TensorLike]":
    """
    Compute the relative angle from ref_angle_rad to angle_rad in the specified direction

    Args:
        ref_angle_rad: reference angle in radians [float]
        angle_rad: tensor of angles to compute relative angles for, in radians
        direction: If "cw", measure clockwise; if "ccw", measure counter-clockwise
    Returns:
        A RelativeAngleResult containing:
        - relative_angle: Tensor of relative angles [same dimension as 'angle_rad', always positive in range [0, 2π)]
        - wrap_around_flag: Tensor of flags whether the relative angle computation required a wrap-around at multiples of 2π
    """

    two_pi = 2 * np.pi

    # Signed difference between ref and angle, then a single reduction to [0, 2π).
    # We subtract before reducing rather than reducing each operand separately:
    # the subtraction with the python-scalar `ref_angle_rad` keeps `angle_rad`'s
    # dtype (a python scalar does not upcast a numpy/torch float32 array), so the
    # whole computation stays in float32 for float32 inputs. Reducing each operand
    # independently instead promoted `ref_angle_rad % 2π` to float64 while
    # `angle_rad % 2π` stayed float32, so the same value reduced to results ~1 ULP
    # apart; for ref == angle_rad[i] that made the relative angle wrap to ~2π
    # instead of 0 and broke, e.g., the strict-monotonicity check on a structured
    # lidar model's column azimuths whose reference reduces near the ±π boundary.
    # (a - b) mod 2π == (a mod 2π - b mod 2π) mod 2π, so this is exact.
    if direction == "cw":
        # Clockwise: going from ref to angle in CW direction.
        signed_diff = ref_angle_rad - angle_rad
    elif direction == "ccw":
        # Counter-clockwise: going from ref to angle in CCW direction.
        signed_diff = angle_rad - ref_angle_rad
    else:
        raise ValueError(f"Invalid spinning direction: {direction}")

    # Wrap-around: the absolute separation spans at least a full revolution.
    wrap_around_flag = abs(angle_rad - ref_angle_rad) >= two_pi

    return RelativeAngleResult(
        relative_angle_rad=cast(TensorLike, signed_diff % two_pi), wrap_around_flag=wrap_around_flag
    )


def compute_max_angle_with_monotonicity(
    fw_poly_coeffs: np.ndarray,
    max_radius: float,
    newton_iterations: int = 20,
) -> float:
    """Find the maximum angle where a forward polynomial is monotonically increasing
    and its output does not exceed ``max_radius``.

    A forward distortion polynomial maps incidence angles (in radians) to radial
    pixel distances from the principal point (in normalized image units, i.e. pixels
    divided by focal length):

    .. math::
        r(\\theta) = c_0 + c_1\\,\\theta + c_2\\,\\theta^2 + \\cdots + c_n\\,\\theta^n

    where :math:`\\theta` is the angle between the camera ray and the optical axis,
    and :math:`r` is the resulting radial distance in the image plane.  The parameter
    ``max_radius`` is the largest such distance that still falls within the image
    (e.g. the distance from the principal point to the farthest image corner, in the
    same normalized units).

    This function returns

    .. math::
        \\theta^* = \\min(\\theta_{\\text{mono}},\\; \\theta_{\\text{radius}})

    where :math:`\\theta_{\\text{mono}}` is the smallest positive angle at which
    the polynomial ceases to be monotonically increasing, and
    :math:`\\theta_{\\text{radius}}` is the angle at which the polynomial output
    first reaches ``max_radius``.

    **Step 1 -- Monotonicity limit via analytical root-finding:**

    The derivative polynomial is

    .. math::
        r'(\\theta) = c_1 + 2\\,c_2\\,\\theta + 3\\,c_3\\,\\theta^2 + \\cdots + n\\,c_n\\,\\theta^{n-1}

    The monotonicity limit :math:`\\theta_{\\text{mono}}` is the smallest positive
    real root of :math:`r'(\\theta) = 0`.  Roots are found via the companion-matrix
    eigenvalue method (``Polynomial.roots()``), which yields all :math:`n-1` roots of the
    degree-:math:`(n-1)` derivative polynomial.  Only real positive roots are
    retained; the smallest one is the monotonicity limit.  If no such root exists,
    the polynomial is globally monotone and :math:`\\theta_{\\text{mono}} = \\infty`.

    For the OpenCV fisheye model the forward polynomial is degree 9 with only odd
    terms, so its derivative is degree 8 with only even terms.  Substituting
    :math:`u = \\theta^2` reduces this to a degree-4 polynomial in :math:`u`, making
    the eigenvalue problem a 4x4 (or equivalently 8x8 in ``Polynomial.roots()`` without the
    substitution -- both are negligible cost).

    **Step 2 -- Deciding the limiting condition:**

    If :math:`r(\\theta_{\\text{mono}}) \\le` ``max_radius``, the polynomial folds
    before reaching the image boundary, and :math:`\\theta_{\\text{mono}}` is returned
    directly.

    **Step 3 -- Newton-Raphson inversion (image-boundary case):**

    Otherwise we solve :math:`r(\\theta) = ` ``max_radius`` via Newton-Raphson:

    .. math::
        \\theta_{k+1} = \\theta_k - \\frac{r(\\theta_k) - r_{\\text{max}}}{r'(\\theta_k)}

    The initial guess is :math:`\\theta_0 = \\min(r_{\\text{max}},\\;
    (1-\\epsilon)\\,\\theta_{\\text{mono}})` where :math:`\\epsilon` is a small
    safety margin.  This is safe because for near-equidistant lenses
    :math:`r \\approx \\theta`, so ``max_radius`` is a good approximation of the
    solution, and we are guaranteed to start within the monotone region.  Each iterate
    is clamped to :math:`(1-\\epsilon)\\,\\theta_{\\text{mono}}` to prevent the solver
    from crossing into the non-monotone region where the Newton step would diverge.
    Convergence is typically achieved in 3-5 iterations (quadratic convergence of
    Newton's method on a smooth monotone function with non-vanishing derivative).

    Note: internal computation is done in float64 for numerical stability of the
    root-finding and Newton iteration, regardless of the input dtype.

    Parameters
    ----------
    fw_poly_coeffs : np.ndarray
        Forward polynomial coefficients ``[c0, c1, c2, ...]`` in ascending order
        of degree, mapping incidence angle (rad) to radial pixel distance
        (normalized by focal length).
    max_radius : float
        Maximum radial pixel distance from the principal point to the image
        boundary (in the same normalized units as the polynomial output).
    newton_iterations : int
        Maximum number of Newton-Raphson iterations for the inversion step.

    Returns
    -------
    float
        Maximum valid angle in radians.
    """
    # Safety margin to keep Newton iterates strictly inside the monotone region.
    # The exact value is not critical -- it only needs to prevent the solver from
    # landing exactly on the root of r'(theta)=0 where the Newton step is undefined.
    _MONO_MARGIN = 1e-3

    # float64 is required for the companion-matrix eigenvalue computation in
    # Polynomial.roots() -- the eigenvalue solver subtracts nearly-equal values
    # and needs the extra precision to distinguish real roots from complex ones.
    coeffs = np.asarray(fw_poly_coeffs, dtype=np.float64)
    degree = len(coeffs) - 1
    if degree < 1:
        return 0.0

    # Build the forward polynomial and its derivative using the modern numpy API
    fw_poly = Polynomial(coeffs)
    d_poly = fw_poly.deriv()

    # Step 1: Find monotonicity limit -- smallest positive real root of r'(theta) = 0
    # Polynomial.roots() computes eigenvalues of the companion matrix, yielding all
    # (degree-1) roots of the derivative polynomial simultaneously.
    monotonicity_limit = np.inf
    if degree >= 2:
        roots = d_poly.roots()
        # Filter to real positive roots (imaginary part below tolerance)
        for root in roots:
            if np.isreal(root) and root.real > 0.0:
                monotonicity_limit = min(monotonicity_limit, root.real)

    # Step 2: Check if the polynomial folds before reaching max_radius
    r_at_limit = float(fw_poly(monotonicity_limit)) if np.isfinite(monotonicity_limit) else np.inf
    if r_at_limit <= max_radius:
        return float(monotonicity_limit)

    # Step 3: Newton-Raphson inversion of r(theta) = max_radius
    mono_clamp = float(monotonicity_limit) * (1.0 - _MONO_MARGIN)
    theta = min(float(max_radius), mono_clamp)
    for _ in range(newton_iterations):
        r = float(fw_poly(theta))
        dr = float(d_poly(theta))
        if abs(dr) < 1e-12:
            break
        theta = theta - (r - max_radius) / dr
        # Clamp to stay strictly within the monotone region
        theta = min(theta, mono_clamp)
        if theta < 0.0:
            theta = 0.0
        if abs(r - max_radius) < 1e-10:
            break

    return float(theta)
