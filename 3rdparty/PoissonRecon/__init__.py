# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import os
import pathlib


def reconstruct_surface(
    input_file: str, output_file: str, width: float = 0.1, density: bool = True, samples_per_node: float = 1.0
):
    """Runs Poisson surface extraction executable on a given input file, producing a reconstruction mesh as output"""

    # find executable based on context
    if (path := pathlib.Path("external/PoissonRecon/poisson_recon")).exists():
        # bazel-build setup
        pass
    elif (path := pathlib.Path(__file__).parent.parent / "poisson_recon").exists():
        # wheel setup
        pass
    else:
        raise FileNotFoundError("PoissonRecon executable not found")

    # run executable
    abs_path = str(path.absolute())
    command = (
        f"'{abs_path}' --in {input_file} --out {output_file} "
        f"--width {width} --density --samplesPerNode {samples_per_node} --colors"
    )
    if density:
        command += " --density"
    os.system(command)
