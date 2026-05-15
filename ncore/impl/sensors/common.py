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

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Union, cast

import numpy as np
import torch


class BaseModel(torch.nn.Module):
    """Base class for all sensor models in ncore"""

    init_device: torch.device  #: Device the model was initialized with
    init_dtype: torch.dtype  #: Floating point dtype the model was initialized with

    def __init__(
        self,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ):
        # Initialize nn.module
        super().__init__()

        # Make sure device is a torch device
        if isinstance(device, str):
            device = torch.device(device)

        self.init_device: torch.device = device

        # Make sure dtype is a torch floating point dtype
        if not dtype.is_floating_point:
            raise TypeError(f"Expected floating point dtype, but got {dtype}")

        self.init_dtype: torch.dtype = dtype

    @property
    def device(self) -> torch.device:
        """Returns the device of the model given by the first parameter or buffer
        with a fallback to the init-time device"""
        try:
            # grab the first parameter or buffer to determine current device
            return next(
                self.parameters(),
                next(self.buffers()),  # if no parameters, grab a buffer
            ).device
        except StopIteration:
            return self.init_device  # otherwise fall back to init

    @property
    def dtype(self) -> torch.dtype:
        """Returns the dtype of the model given by the first floating point dtype of parameters / buffers
        with a fallback to the init-time dtype"""

        # check parameters first
        for parameter in self.parameters():
            if not parameter.dtype.is_floating_point:
                continue
            return parameter.dtype
        # if no parameters, grab a buffer
        for buffer in self.buffers():
            if not buffer.dtype.is_floating_point:
                continue
            return buffer.dtype
        # otherwise fall back to init
        return self.init_dtype


def to_torch(
    var: Union[torch.Tensor, np.ndarray],
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Converts an input array / tensor to a tensor on the target device (with optional dtype conversion)."""
    if isinstance(var, np.ndarray):
        # Torch doesn't support uint32 and uint64 so we cast them to signed integers beforehand
        # Note that this can cause problems

        var = cast(np.ndarray, var)

        if var.dtype == np.uint16:
            assert np.all(var <= np.iinfo(np.int16).max), (
                "[CameraModel]: Trying to cast uint16 to int16 but the value exceeds max range."
            )
            var = var.astype(np.int16)

        if var.dtype == np.uint32:
            assert np.all(var <= np.iinfo(np.int32).max), (
                "[CameraModel]: Trying to cast uint32 to int32 but the value exceeds max range."
            )
            var = var.astype(np.int32)

        if var.dtype == np.uint64:
            assert np.all(var <= np.iinfo(np.int64).max), (
                "[CameraModel]: Trying to cast uint64 to int64 but the value exceeds max range."
            )
            var = var.astype(np.int64)

        var = torch.from_numpy(var)

    return var.to(device=device, dtype=dtype)


def eval_poly_horner(poly_coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluates a polynomial y=f(x) (given by poly_coefficients) at points x using
    numerically stable Horner scheme"""

    y = torch.zeros_like(x)
    for fi in torch.flip(poly_coefficients, dims=(0,)):
        y = y * x + fi

    return y


def eval_poly_inverse_horner_newton(
    poly_coefficients: torch.Tensor,
    poly_derivative_coefficients: torch.Tensor,
    inverse_poly_approximation_coefficients: torch.Tensor,
    newton_iterations: int,
    y: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the inverse x = f^{-1}(y) of a reference polynomial y=f(x) (given by poly_coefficients) at points y
    using numerically stable Horner scheme and Newton iterations starting from an approximate solution \\hat{x} = \\hat{f}^{-1}(y)
    (given by inverse_poly_approximation_coefficients) and the polynomials derivative df/dx (given by poly_derivative_coefficients)
    """

    x = eval_poly_horner(
        inverse_poly_approximation_coefficients, y
    )  # approximation / starting points - also returned for zero iterations
    assert newton_iterations >= 0, "Newton-iteration number needs to be non-negative"

    # Buffers of intermediate results to allow differentiation
    # (only allocate entries as needed during iteration)
    x_iter = [x]

    for i in range(newton_iterations):
        # Evaluate single Newton step
        dfdx = eval_poly_horner(poly_derivative_coefficients, x_iter[i])
        residuals = eval_poly_horner(poly_coefficients, x_iter[i]) - y
        x_iter.append(x_iter[i] - residuals / dfdx)

    return x_iter[newton_iterations]


def rotmat_to_unitquat(R: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of rotation matrices to unit quaternion representation.

    Args:
        R: batch of rotation matrices [bs, 3, 3]

    Returns:
        batch of unit quaternions (XYZW convention)  [bs, 4]
    """

    num_rotations, D1, D2 = R.shape
    assert (D1, D2) == (3, 3), "Input has to be a Bx3x3 tensor."

    decision_matrix = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)
    quat = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)

    decision_matrix[:, :3] = R.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(dim=1)
    choices = decision_matrix.argmax(dim=1)

    ind = torch.nonzero(choices != 3, as_tuple=True)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * R[ind, i, i]
    quat[ind, j] = R[ind, j, i] + R[ind, i, j]
    quat[ind, k] = R[ind, k, i] + R[ind, i, k]
    quat[ind, 3] = R[ind, k, j] - R[ind, j, k]

    ind = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat[ind, 0] = R[ind, 2, 1] - R[ind, 1, 2]
    quat[ind, 1] = R[ind, 0, 2] - R[ind, 2, 0]
    quat[ind, 2] = R[ind, 1, 0] - R[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat = quat / torch.norm(quat, dim=1)[:, None]

    return quat


def unitquat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of unit quaternions into a SO3 representation.
    Args:
        quat: batch of unit quaternions (XYZW convention) [bs, 4]

    Returns:
        batch of SO3 rotation matrices [bs, 3, 3]
    """

    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]

    # Pre-compute squared terms to avoid redundant computation (each was previously computed 3 times)
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    R = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)

    R[..., 0, 0] = x2 - y2 - z2 + w2
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 2, 0] = 2 * (x * z - y * w)

    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 1, 1] = -x2 + y2 - z2 + w2
    R[..., 2, 1] = 2 * (y * z + x * w)

    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 2] = -x2 - y2 + z2 + w2

    return R


def unitquat_slerp(quat_s: torch.Tensor, quat_e: torch.Tensor, t: torch.Tensor, shortest_arc=True) -> torch.Tensor:
    """
    Batch-wise implementation of SLERP (spherical linear interpolation)

    Args:
        quat_s: batch of unit quaternions denoting the start rotation [bs, 4]
        quat_e: batch of unit quaternions denoting the end rotation  [bs, 4]
        t: interpolation steps within 0.0 and 1.0, 0.0 corresponding to q0 and 1.0 to q1 [bs]
        shortest_arc: if True, interpolation will be performed along the shortest arc on SO(3)
    Returns:
        batch of interpolated quaternions [bs, 4]
    """

    assert quat_s.shape == quat_e.shape, "Input quaternions must be of the same shape."

    if len(quat_s.shape) == 1:
        quat_s = torch.unsqueeze(quat_s, 0)
        quat_e = torch.unsqueeze(quat_e, 0)

    assert t.ndim == 1 and t.shape[0] == quat_e.shape[0], "t is expected to have shape [bs]."

    # omega is the 'angle' between both quaternions
    cos_omega = torch.sum(quat_s * quat_e, dim=-1)

    if shortest_arc:
        # Flip quaternions with negative angle to perform shortest arc interpolation.
        quat_e = torch.where((cos_omega < 0).unsqueeze(-1), -quat_e, quat_e)
        cos_omega = torch.abs(cos_omega)

    # True when q0 and q1 are close.
    nearby_quaternions = cos_omega > (1.0 - 1e-3)

    # Clamp to avoid numerical issues in acos at backward pass, as the derivative of
    # acos is undefined at 1 and -1.
    cos_omega = torch.clamp(cos_omega, -1.0 + 1e-6, 1.0 - 1e-6)

    # General approach
    omega = torch.acos(cos_omega)
    alpha = torch.sin((1 - t) * omega)

    beta = torch.sin(t * omega)
    # Use linear interpolation for nearby quaternions
    alpha = torch.where(nearby_quaternions, (1 - t), alpha)
    beta = torch.where(nearby_quaternions, t, beta)

    # Interpolation
    quat = alpha.reshape(-1, 1) * quat_s + beta.reshape(-1, 1) * quat_e
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    return quat


# --- Rolling-shutter iteration utilities ---


class RollingShutterSolver:
    """Generic iterative solver for rolling-shutter sensors.

    Encapsulates the full rolling-shutter projection workflow: projects world points
    from both start and end poses to determine validity, then iteratively refines
    relative frame time assignments until convergence.

    Mathematical Formulation
    ------------------------
    The rolling-shutter problem is a fixed-point equation:

        t = g(t)

    where ``t`` is the per-point relative frame time in [0, 1] and ``g`` is the
    composition:

        g(t) = relative_frame_time(project(interpolate_pose(t) @ world_point))

    The solver starts from an initial estimate (projection from start/end pose) and
    iterates until convergence, measured by mean absolute change in ``t``.

    Why Picard Converges Rapidly
    ----------------------------
    For standard rolling-shutter sensors (cameras with row-based readout, spinning
    lidars with column-based readout), ``g(t)`` is nearly constant -- its derivative
    ``g'(t*)`` at the fixed point is close to zero (typically 0.0001 to 0.02 for AV
    motions). This is because the sensor readout is much faster than scene motion:
    a small change in interpolation time ``t`` barely changes which row/column the
    point projects to.

    In this regime (``|g'| << 1``), Picard iteration ``t_{k+1} = g(t_k)`` is
    mathematically equivalent to Newton's method on the residual ``f(t) = g(t) - t``:

        Newton: t_{k+1} = t_k - f(t_k) / f'(t_k) = t_k - (g(t_k) - t_k) / (g'(t_k) - 1)

    When ``g' ~ 0``, the denominator ``g' - 1 ~ -1``, giving:

        t_{k+1} ~ t_k + (g(t_k) - t_k) = g(t_k)   (= Picard)

    This explains why Picard converges in just 1-3 iterations: it already achieves
    Newton-like convergence without computing any derivatives.

    The slope ``g'`` can be estimated cheaply from the two endpoint projections:

        g'_est = (g(1) - g(0)) / 1 = t_at_end_pose - t_at_start_pose

    For the tested AV scenarios this is always < 0.02, confirming the near-zero
    slope assumption.

    When to Use Acceleration Methods
    ---------------------------------
    The Secant and Aitken methods are available for sensors where ``g(t)`` has a
    steeper slope (``|g'|`` closer to 1), which would slow Picard convergence:

    - **Non-linear time progressions**: Cameras with undistorted rolling-shutter
      readout (e.g., variable scan rate, non-uniform row timing) where the mapping
      from projected coordinate to time is non-linear and potentially steeper.
    - **Continuous time sensors**: Event cameras or line-scan sensors where time is
      a smooth (not quantized) function of the projection coordinate.
    - **Very tight thresholds**: When ``stop_t_delta < 1e-6`` is required and Picard
      would need 10+ iterations.

    For standard rolling-shutter cameras and spinning lidars with quantized readout
    (floor/column-based time), the ``floor()`` operation makes ``g(t)`` piecewise
    constant (staircase), which is non-differentiable. The acceleration methods may
    overshoot in this regime. Picard handles the staircase naturally since it only
    requires function evaluation, not smoothness.

    Convergence Methods
    -------------------
    **PICARD** (default, order 1 -- linear):
        Simple substitution: ``t_{k+1} = g(t_k)``.
        Optimal for standard sensors where ``|g'| << 1`` (converges in 1-3 iterations
        for cameras, 5-6 for lidars with rate-based stopping).

    **SECANT** (order ~1.618 -- superlinear):
        Applies the secant method element-wise on the residual ``f(t) = g(t) - t``:
            ``t_{k+1} = t_k - f(t_k) * (t_k - t_{k-1}) / (f(t_k) - f(t_{k-1}))``
        Falls back to Picard on the first iteration (no history available).
        Suitable for continuous sensors with steeper ``g'``.

    **AITKEN** (superlinear -- accelerated linear):
        Accumulates 3 Picard iterates ``t0, t1, t2`` then applies Aitken
        delta-squared extrapolation:
            ``t_acc = t0 - (t1 - t0)^2 / (t2 - 2*t1 + t0)``
        Guards against zero denominators by falling back to ``t2``.
        Restarts the 3-step cycle from ``t_acc``.
        Suitable when Picard converges but slowly (``|g'|`` moderate).

    Convergence is measured by mean absolute change in relative frame time between
    iterations. Empirically, pixel-coordinate convergence and time convergence are
    tightly coupled (roughly quadratic co-convergence), so a single time-based
    threshold is sufficient for both camera and lidar sensors.

    Sensor-specific behaviour is provided by implementing the ``Projector`` protocol
    and passing it to ``solve()``.
    """

    @dataclass
    class ProjectionResult:
        """Result of a single projection step.

        Attributes:
            projected: Projected coordinates (image points or sensor angles) [n, D]
            valid_flag: Boolean mask where True indicates a valid projection [n]
        """

        projected: torch.Tensor
        valid_flag: torch.Tensor

    @dataclass
    class Result:
        """Final result of the rolling-shutter iterative solver.

        Attributes:
            rot_rs: Interpolated rotation matrices [n_final, 3, 3]
            trans_rs: Interpolated translations [n_final, 3]
            t: Relative frame times [n_final]
            projection: Final projected coordinates [n_final, D]
            projection_valid: Boolean mask over all input points indicating final validity [n]
            projection_init: Initial projected values for all input points [n, D]
        """

        rot_rs: torch.Tensor
        trans_rs: torch.Tensor
        t: torch.Tensor
        projection: torch.Tensor
        projection_valid: torch.Tensor
        projection_init: torch.Tensor

    class Projector(Protocol):
        """Protocol for sensor-specific rolling-shutter projection behaviour.

        Implementations provide sensor-specific projection and time computation.
        """

        def project(self, sensor_points: torch.Tensor) -> RollingShutterSolver.ProjectionResult:
            """Project sensor-frame points to sensor-specific coordinates.

            Parameters
            ----------
            sensor_points : torch.Tensor
                Sensor-frame points [n, 3]

            Returns
            -------
            RollingShutterSolver.ProjectionResult
                Projected coordinates [n, D] and validity mask [n]
            """
            ...

        def relative_frame_time(self, projected: torch.Tensor) -> torch.Tensor:
            """Compute relative frame times from projected coordinates.

            Parameters
            ----------
            projected : torch.Tensor
                Projected values (image points or sensor angles) [n, D]

            Returns
            -------
            torch.Tensor
                Relative frame times in [0, 1] range [n]
            """
            ...

    @staticmethod
    def _evaluate_g(
        t: torch.Tensor,
        world_points_valid: torch.Tensor,
        quat_start: torch.Tensor,
        quat_end: torch.Tensor,
        transl_start: torch.Tensor,
        transl_end: torch.Tensor,
        projector: RollingShutterSolver.Projector,
    ) -> tuple[torch.Tensor, RollingShutterSolver.ProjectionResult, torch.Tensor, torch.Tensor]:
        """Evaluate the fixed-point map g(t) for the given time values.

        Returns (g_t, projection_result, rot_rs, trans_rs) where g_t contains
        the new time values for valid projections and zeros elsewhere.
        """
        rot_rs = unitquat_to_rotmat(unitquat_slerp(quat_start, quat_end, t))  # [n_valid, 3, 3]
        trans_rs = (1 - t)[..., None] * transl_start + t[..., None] * transl_end
        sensor_points = (torch.bmm(rot_rs, world_points_valid[:, :, None]) + trans_rs[..., None]).squeeze(-1)
        projection_result = projector.project(sensor_points)

        g_t = torch.zeros_like(t)
        if projection_result.valid_flag.any():
            g_t[projection_result.valid_flag] = projector.relative_frame_time(
                projection_result.projected[projection_result.valid_flag]
            )
        return g_t, projection_result, rot_rs, trans_rs

    @staticmethod
    def solve(
        world_points: torch.Tensor,
        T_world_sensor_start: torch.Tensor,
        T_world_sensor_end: torch.Tensor,
        max_iterations: int,
        stop_t_delta: float,
        projector: RollingShutterSolver.Projector,
        method: Literal["PICARD", "SECANT", "AITKEN"] = "PICARD",
        stop_t_delta_rate: float = 0.0,
    ) -> RollingShutterSolver.Result:
        """Run the iterative rolling-shutter solver.

        Projects world points from both start and end poses to determine validity,
        then iteratively refines relative frame time assignments until convergence,
        measured by mean absolute time change between iterations falling below
        ``stop_t_delta``.

        Parameters
        ----------
        world_points : torch.Tensor
            World points [n, 3]
        T_world_sensor_start : torch.Tensor
            Start-of-frame sensor pose as a 4x4 rigid transformation matrix [4, 4]
        T_world_sensor_end : torch.Tensor
            End-of-frame sensor pose as a 4x4 rigid transformation matrix [4, 4]
        max_iterations : int
            Maximum number of iterations
        stop_t_delta : float
            Convergence threshold: mean absolute change in relative frame time
            between consecutive iterations.

            Default is 1e-4 to give sub-pixel accuracy for cameras (for a 1080-row image, 1e-4 in t-space corresponds to ~0.1 px)
            and to match the original lidar solver's threshold (stop_mean_relative_time_error=1e-4).
            Converges in 1-4 iterations for typical automotive inter-frame motions
            (5-30 deg rotation + 0.1-0.5m translation).

            Use 1e-6 for sub-millipixel precision (requires ~10-20 iterations with
            Picard, or fewer with acceleration methods).
        projector : RollingShutterSolver.Projector
            Protocol implementation providing sensor-specific projection behaviour
        method : ConvergenceMethod
            Convergence acceleration method (default: PICARD).
            See class docstring for details on each method.
        stop_t_delta_rate : float
            Secondary stopping criterion: stop when the *change* in t_delta between
            consecutive iterations falls below this threshold (diminishing returns).
            This handles cases where the solver is making progress but slowly
            (e.g., lidar with discrete column quantization). Set to 0.0 to disable.

        Returns
        -------
        RollingShutterSolver.Result
            Final interpolated poses, times, projection results, validity mask,
            and initial projections for all input points.
        """
        # Project all world points from start and end poses
        sensor_points_start = (
            T_world_sensor_start[:3, :3] @ world_points.T + T_world_sensor_start[:3, 3, None]
        ).T  # [n, 3]
        proj_start = projector.project(sensor_points_start)

        sensor_points_end = (T_world_sensor_end[:3, :3] @ world_points.T + T_world_sensor_end[:3, 3, None]).T  # [n, 3]
        proj_end = projector.project(sensor_points_end)

        # Determine validity: a point is a candidate if it projects validly from
        # either the start or end pose. Initial projection prefers start-of-frame.
        #
        # NOTE: This union-of-endpoints heuristic can theoretically miss points that
        # are outside the FOV at both t=0 and t=1 but inside at some intermediate t
        # (the sensor "sweeps past" the point during rotation). Analysis shows this
        # gap is negligible for typical AV inter-frame motions:
        #   - At 5 deg/frame rotation: affects < 0.08 px at FOV edge
        #   - At 10 deg/frame: < 0.33 px
        #   - Requires > 17 deg/frame before 1 full pixel is affected
        # For spinning lidar (360-deg azimuth), this gap does not exist in practice.
        # If needed for extreme rotations, check validity at t=0.5 as well.
        valid = proj_start.valid_flag | proj_end.valid_flag
        projected_init = proj_end.projected.clone()
        projected_init[proj_start.valid_flag] = proj_start.projected[proj_start.valid_flag]

        # Filter to valid points for iteration
        world_points_valid = world_points[valid, :]
        n_valid = world_points_valid.shape[0]

        # Early return if no valid points
        if n_valid == 0:
            d = projected_init.shape[1] if projected_init.ndim > 1 else 0
            return RollingShutterSolver.Result(
                rot_rs=torch.empty((0, 3, 3), dtype=world_points.dtype, device=world_points.device),
                trans_rs=torch.empty((0, 3), dtype=world_points.dtype, device=world_points.device),
                t=torch.empty(0, dtype=world_points.dtype, device=world_points.device),
                projection=torch.empty((0, d), dtype=world_points.dtype, device=world_points.device),
                projection_valid=valid,
                projection_init=projected_init,
            )

        # Convert rotation matrices to quaternions for slerp interpolation
        quat_start = rotmat_to_unitquat(T_world_sensor_start[None, :3, :3]).expand(n_valid, -1)  # [n_valid, 4]
        quat_end = rotmat_to_unitquat(T_world_sensor_end[None, :3, :3]).expand(n_valid, -1)  # [n_valid, 4]
        transl_start = T_world_sensor_start[:3, 3]  # [3]
        transl_end = T_world_sensor_end[:3, 3]  # [3]

        # Derive initial times from the initial projected values
        t = projector.relative_frame_time(projected_init[valid])

        rot_rs = torch.empty(0, dtype=world_points.dtype, device=world_points.device)
        trans_rs = torch.empty(0, dtype=world_points.dtype, device=world_points.device)
        projection_result = RollingShutterSolver.ProjectionResult(
            projected=torch.empty(0, dtype=world_points.dtype, device=world_points.device),
            valid_flag=torch.empty(0, dtype=torch.bool, device=world_points.device),
        )

        # --- Method-specific state initialization ---
        # Secant method: previous t and f(t) values
        t_prev: Optional[torch.Tensor] = None
        f_prev: Optional[torch.Tensor] = None

        # Aitken method: accumulator for 3-step Picard sub-cycles
        aitken_step = 0  # counts 0, 1, 2 within each Aitken cycle
        aitken_t0: Optional[torch.Tensor] = None
        aitken_t1: Optional[torch.Tensor] = None

        # Delta-rate tracking: previous t_delta for diminishing-returns criterion
        prev_t_delta = float("inf")

        for _ in range(max_iterations):
            # Evaluate the fixed-point map g(t)
            g_t, projection_result, rot_rs, trans_rs = RollingShutterSolver._evaluate_g(
                t, world_points_valid, quat_start, quat_end, transl_start, transl_end, projector
            )

            if not projection_result.valid_flag.any():
                break

            valid_mask = projection_result.valid_flag
            g_valid = g_t[valid_mask]  # g(t) for valid points
            t_valid = t[valid_mask]  # current t for valid points

            # --- Apply convergence method ---
            if method == "PICARD":
                t_new = g_valid

            elif method == "SECANT":
                f_curr = g_valid - t_valid  # residual f(t) = g(t) - t

                if t_prev is not None and f_prev is not None:
                    # Full secant update (element-wise)
                    t_prev_valid = t_prev[valid_mask]
                    f_prev_valid = f_prev[valid_mask]
                    denom = f_curr - f_prev_valid

                    # Where denominator is too small, fall back to Picard
                    safe = denom.abs() > 1e-12
                    t_new = torch.where(
                        safe,
                        t_valid - f_curr * (t_valid - t_prev_valid) / denom,
                        g_valid,  # Picard fallback
                    )
                else:
                    # First iteration: no history, use Picard
                    t_new = g_valid

                # Store history for next iteration (full-sized tensors)
                t_prev = t.clone()
                f_prev = torch.zeros_like(t)
                f_prev[valid_mask] = f_curr

            elif method == "AITKEN":
                if aitken_step == 0:
                    aitken_t0 = t.clone()
                    t_new = g_valid  # Picard step 1
                    aitken_step = 1
                elif aitken_step == 1:
                    aitken_t1 = t.clone()
                    aitken_t1[valid_mask] = g_valid  # store t1 with the Picard update applied
                    t_new = g_valid  # Picard step 2
                    aitken_step = 2
                else:
                    # aitken_step == 2: we have t0, t1, and now t2 = g(t1)
                    assert aitken_t0 is not None and aitken_t1 is not None
                    t2_valid = g_valid
                    t0_valid = aitken_t0[valid_mask]
                    t1_valid = aitken_t1[valid_mask]

                    # Aitken delta-squared extrapolation
                    denom = t2_valid - 2.0 * t1_valid + t0_valid
                    numerator = (t1_valid - t0_valid) ** 2
                    safe = denom.abs() > 1e-12

                    t_acc = torch.where(
                        safe,
                        t0_valid - numerator / denom,
                        t2_valid,  # fallback to latest Picard iterate
                    )
                    t_new = t_acc

                    # Restart the 3-step cycle
                    aitken_step = 0
                    aitken_t0 = None
                    aitken_t1 = None

            else:
                raise ValueError(f"Unknown convergence method: {method}")

            # Compute convergence criterion and update t
            t_delta = (t_new - t[valid_mask]).abs().mean().item()
            t[valid_mask] = t_new

            if t_delta < stop_t_delta:
                break

            # Delta-rate criterion: stop when progress stalls (diminishing returns)
            if stop_t_delta_rate > 0.0 and abs(prev_t_delta - t_delta) <= stop_t_delta_rate:
                break
            prev_t_delta = t_delta

        # Combine initial validity (projected from start/end pose) with final iteration validity
        # into a single [n] mask over all input points.
        # Clone is required: we use `valid` as the boolean index while mutating the copy.
        projection_valid = valid.clone()
        projection_valid[valid] = projection_result.valid_flag

        # Filter outputs to only truly-valid points (valid AND final-iteration-valid)
        final_valid = projection_result.valid_flag
        return RollingShutterSolver.Result(
            rot_rs=rot_rs[final_valid],
            trans_rs=trans_rs[final_valid],
            t=t[final_valid],
            projection=projection_result.projected[final_valid],
            projection_valid=projection_valid,
            projection_init=projected_init,
        )
