# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import pathlib


try:
    from python.runfiles import Runfiles

    # bazel-build-setup
    _RUNFILES = Runfiles.Create()
except ImportError:
    # wheel setup
    _RUNFILES = None


def reconstruct_surface(
    input_file: str, output_file: str, width: float = 0.1, density: bool = True, samples_per_node: float = 1.0
):
    """Runs Poisson surface extraction executable on a given input file, producing a reconstruction mesh as output"""

    # find executable based on context
    if _RUNFILES is not None:
        # bazel-build setup
        assert isinstance(rlocation := _RUNFILES.Rlocation("poisson_recon/poisson_recon_bin"), str)
        path = pathlib.Path(rlocation)
    elif (path := pathlib.Path(__file__).parent.parent / "poisson_recon_bin").exists():
        # wheel setup
        pass
    else:
        raise FileNotFoundError("poisson_recon_bin executable not found")

    # run executable
    abs_path = str(path.absolute())
    command = (
        f"'{abs_path}' --in {input_file} --out {output_file} "
        f"--width {width} --density --samplesPerNode {samples_per_node} --colors"
    )
    if density:
        command += " --density"
    os.system(command)
