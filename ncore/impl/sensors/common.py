# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

from typing import Union, Optional

import torch
import numpy as np


def to_torch(
    var: Union[torch.Tensor, np.ndarray],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Converts an input array / tensor to a tensor on the target device (with optional dtype conversion)."""
    if isinstance(var, np.ndarray):
        # Torch doesn't support uint32 and uint64 so we cast them to signed integers beforehand
        # Note that this can cause problems

        if var.dtype == np.uint16:
            assert np.all(
                var <= np.iinfo(np.int16).max
            ), "[CameraModel]: Trying to cast uint16 to int16 but the value exceeds max range."
            var = var.astype(np.int16)

        if var.dtype == np.uint32:
            assert np.all(
                var <= np.iinfo(np.int32).max
            ), "[CameraModel]: Trying to cast uint32 to int32 but the value exceeds max range."
            var = var.astype(np.int32)

        if var.dtype == np.uint64:
            assert np.all(
                var <= np.iinfo(np.int64).max
            ), "[CameraModel]: Trying to cast uint64 to int64 but the value exceeds max range."
            var = var.astype(np.int64)

        var = torch.from_numpy(var)

    return var.to(device=device, dtype=dtype)


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

    R = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)

    R[..., 0, 0] = torch.pow(x, 2) - torch.pow(y, 2) - torch.pow(z, 2) + torch.pow(w, 2)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 2, 0] = 2 * (x * z - y * w)

    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 1, 1] = -torch.pow(x, 2) + torch.pow(y, 2) - torch.pow(z, 2) + torch.pow(w, 2)
    R[..., 2, 1] = 2 * (y * z + x * w)

    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 2] = -torch.pow(x, 2) - torch.pow(y, 2) + torch.pow(z, 2) + torch.pow(w, 2)

    return R


def unitquat_slerp(quat_s: torch.Tensor, quat_e: torch.Tensor, t: torch.Tensor, shortest_arc=True) -> torch.Tensor:
    """
    Batch-wise implementation of SLERP (spherical linear interpolation)

    Args:
        quat_s: batch of unit quaternions denoting the start rotation [bs, 4]
        quat_e: batch of unit quaternions denoting the end rotation  [bs, 4]
        t: interpolation steps within 0.0 and 1.0, 0.0 corresponding to q0 and 1.0 to q1 [bs, 1]
        shortest_arc: if True, interpolation will be performed along the shortest arc on SO(3)
    Returns:
        batch of interpolated quaternions [bs, 4]
    """

    assert quat_s.shape == quat_e.shape, "Input quaternions must be of the same shape."

    if len(quat_s.shape) == 1:
        quat_s = torch.unsqueeze(quat_s, 0)
        quat_e = torch.unsqueeze(quat_e, 0)

    # omega is the 'angle' between both quaternions
    cos_omega = torch.sum(quat_s * quat_e, dim=-1)

    if shortest_arc:
        # Flip quaternions with negative angle to perform shortest arc interpolation.
        quat_e = quat_e.clone()
        quat_e[cos_omega < 0, :] *= -1
        cos_omega = torch.abs(cos_omega)

    # True when q0 and q1 are close.
    nearby_quaternions = cos_omega > (1.0 - 1e-3)

    # General approach
    omega = torch.acos(cos_omega)
    alpha = torch.sin((1 - t) * omega)

    beta = torch.sin(t * omega)
    # Use linear interpolation for nearby quaternions
    alpha[nearby_quaternions] = (1 - t)[nearby_quaternions]
    beta[nearby_quaternions] = t[nearby_quaternions]

    # Interpolation
    quat = alpha.reshape(-1, 1) * quat_s + beta.reshape(-1, 1) * quat_e
    quat /= torch.norm(quat, dim=-1, keepdim=True)

    return quat
