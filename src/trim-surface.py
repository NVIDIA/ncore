import argparse

import numpy as np
import point_cloud_utils as pcu


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_points", type=str)
    argparser.add_argument("mesh", type=str)
    argparser.add_argument("trim_distance", type=float,
                           help="Trim vertices of the reconstructed mesh whose nearest "
                                "point in the input is greater than this value. The units of this argument are voxels "
                                "(where the cells_per_axis determines the size of a voxel) Default is -1.0.")
    argparser.add_argument("--scale", type=float, default=1.1,
                           help="Pad the bounding box of the input point cloud by a factor if --scale. "
                                "i.e. the the diameter of the padded bounding box is --scale times bigger than the "
                                "diameter of the bounding box of the input points. Defaults is 1.1.")
    argparser.add_argument("--out", type=str, default="trimmed.ply", help="Path to file to save trim mesh to.")
    args = argparser.parse_args()

    print(f"Loading input point cloud {args.input_points}")
    p = pcu.load_mesh_v(args.input_points)

    print(f"Loading reconstructed mesh {args.mesh}")
    v, f, n, c = pcu.load_mesh_vfnc(args.mesh)

    print(f"There are {p.shape[0]} points and the mesh has {v.shape[0]} vertices and {f.shape[0]} faces")

    print("Trimming mesh...")
    # Trim distance in world coordinates
    trim_dist_world = args.trim_distance
    print("Finding nearest neighbor")
    nn_dist, _ = pcu.k_nearest_neighbors(v, p, k=2)
    nn_dist = nn_dist[:, 1]
    print("Stacking faces")
    f_mask = np.stack([nn_dist[f[:, i]] < trim_dist_world for i in range(f.shape[1])], axis=-1)
    print("Filtering faces")
    f_mask = np.all(f_mask, axis=-1)
    f = f[f_mask]

    print("Saving trimmed mesh...")
    pcu.save_mesh_vfnc(args.out, v, f, n, c)
    print("Done!")


if __name__ == "__main__":
    main()
