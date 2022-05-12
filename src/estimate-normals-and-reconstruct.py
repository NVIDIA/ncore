import argparse
import os

import numpy as np
import point_cloud_utils as pcu
import tqdm
import struct


#
# TODO: @Zan, you need to replace this function with the thing that loads your track data,
# then everything should work. Deps are just the imports. I finally got the namespace
# for point-cloud-utils so you need to `pip install point-cloud-utils` to get the latest.
# 
def load_dat_rays(dat_file_path, split_arrays=True):
    if dat_file_path.endswith('.npy'):
        lidar_dat = np.load(dat_file_path)
    else:
        # with open(dat_file_path, 'rb') as f:
        #     # The first number denotes the number of points
        #     d = f.read(4)
        #     n_pts = struct.unpack('>i', d)[0]
        #     lidar_dat = np.array(struct.unpack('=%sf' % n_pts, f.read())).reshape([-1, 9])
        with open(dat_file_path, 'rb') as f:
            # The first number denotes the number of points
            n_rows, n_columns = struct.unpack('<ii', f.read(8))

            # The remaining data are floats saved in little endian
            # Columns contain: x_s, y_s, z_s, x_e, y_e, z_e, d, intensity, dynamic_flag
            lidar_dat = np.array(struct.unpack('<%sf' % (n_rows * n_columns), f.read())).reshape(n_rows, n_columns)

    if lidar_dat.shape[-1] == 9:
        ray_origin, ray_end, ray_dist, intensity, dynamic_flag = \
            lidar_dat[:, 0:3], lidar_dat[:, 3:6], lidar_dat[:, 6], lidar_dat[:, 7], lidar_dat[:, 8]
        rgb = np.zeros_like(ray_origin)
    elif lidar_dat.shape == 12:  # Has color
        ray_origin, ray_end, ray_dist, intensity, dynamic_flag, rgb = \
            lidar_dat[:, 0:3], lidar_dat[:, 3:6], lidar_dat[:, 6], lidar_dat[:, 7], lidar_dat[:, 8], lidar_dat[:, 9:]
    else:
        raise ValueError("Bad data file should have shape (9, N) or (12, N)")

    ray_dir = ray_end - ray_origin
    ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=-1, keepdims=True)
    rgb = np.zeros_like(ray_origin)

    if split_arrays:
        return ray_origin, ray_dir, ray_dist, intensity, dynamic_flag, rgb
    else:
        return lidar_dat



def read_rays(file_path, ignore_color=True, min_dist=2.75):
    ray_origin, ray_dir, ray_dist, intensity, dynamic_flag, rgb = load_dat_rays(file_path)
    dynamic_flag = dynamic_flag > 0.0
    if ignore_color:
        mask = np.logical_and(ray_dist >= min_dist, ~dynamic_flag)
    else:
        mask = np.logical_and(np.logical_and(ray_dist >= min_dist, rgb[:, 0] > 0.0), ~dynamic_flag)
        mask = np.logical_and(intensity > 0.0, mask)
    pts = ray_origin[mask] + ray_dir[mask] * ray_dist[mask][:, np.newaxis]
    clr = rgb[mask] / 255.0
    dist = ray_dist[mask]
    dirs = ray_dir[mask]
    return pts.astype(np.float16), dirs.astype(np.float16), dist.astype(np.float16), clr.astype(np.float16)


def load_fused_pc(path, max_dist=np.inf, start_idx=0, end_idx=-1, ignore_color=False):
    files = sorted([os.path.join(path, fname) for fname in os.listdir(path)
                    if fname.endswith(".dat") or fname.endswith(".npy")])
    pts = []
    dirs = []
    clrs = [] if not ignore_color else None
    if end_idx > 0:
        files = files[start_idx:end_idx]
    pbar = tqdm.tqdm(files)
    for fname in pbar:
        pbar.set_postfix({'filename': fname})
        ray_hit, ray_dir, ray_dist, ray_clr = read_rays(fname, ignore_color=ignore_color)
        if np.isfinite(max_dist):
            dist_mask = ray_dist < max_dist
            ray_hit, ray_dir, ray_dist, ray_clr = ray_hit[dist_mask], ray_dir[dist_mask], \
                                                  ray_dist[dist_mask], ray_clr[dist_mask]
        pts.append(ray_hit)
        dirs.append(-ray_dir)
        if not ignore_color:
            clrs.append(ray_clr)
    pts = np.concatenate(pts)
    dirs = np.concatenate(dirs)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    # import torch
    # torch.save((pts, dirs), "out.pth")
    if not ignore_color:
        clrs = np.concatenate(clrs)

    return pts, dirs, clrs


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("lidar_dat_path", type=str)
    argparser.add_argument("radius", type=int)
    argparser.add_argument("--mode", type=str, default='knn', help="one of 'knn', 'ball' or 'ball-rbf'")
    argparser.add_argument("--min-angle-thresh", type=float, default=90.0)
    argparser.add_argument("--max-dist", type=float, default=np.inf)
    argparser.add_argument("--start-at", type=int, default=0)
    argparser.add_argument("--stop-at", type=int, default=-1)
    argparser.add_argument("--ignore-color", action="store_true")
    argparser.add_argument("--spsr-path", type=str, default="./external/AdaptiveSolvers/Bin/Linux/PoissonRecon",
                           help="Path to binary for Screened Poisson Reconstruction")
    cmd_args = argparser.parse_args()

    if cmd_args.mode not in ['knn', 'ball', 'ball-rbf']:
        raise ValueError(f"Invalid --mode, must be one of 'knn', 'ball', 'ball-rbf' but got '{cmd_args.mode}'")

    print("Loading points and ray directions...")
    p, d, c = load_fused_pc(cmd_args.lidar_dat_path,
                            max_dist=cmd_args.max_dist,
                            start_idx=cmd_args.start_at,
                            end_idx=cmd_args.stop_at,
                            ignore_color=cmd_args.ignore_color)
    print(f"Loaded {p.shape[0]} points.")

    print("Converting to float32...")
    p, d = p.astype(np.float32), d.astype(np.float32)
    np.savez("points_directions.npz", p=p, d=d)

    print("Estimating normals...")
    if cmd_args.mode == 'knn':
        pid, nf = pcu.estimate_point_cloud_normals_knn(p, cmd_args.radius, view_directions=d,
                                                       drop_angle_threshold=np.deg2rad(cmd_args.min_angle_thresh))
    elif cmd_args.mode == 'ball':
        pid, nf = pcu.estimate_point_cloud_normals_ball(p, cmd_args.radius, view_directions=d,
                                                        weight_function="constant",
                                                        drop_angle_threshold=np.deg2rad(cmd_args.min_angle_thresh))
    elif cmd_args.mode == 'ball-rbf':
        pid, nf = pcu.estimate_point_cloud_normals_ball(p, cmd_args.radius, view_directions=d,
                                                        weight_function="rbf",
                                                        drop_angle_threshold=np.deg2rad(cmd_args.min_angle_thresh))
    else:
        raise ValueError(f"Invalid --mode, must be one of 'knn', 'ball', 'ball-rbf' but got '{cmd_args.mode}'")

    pf = p[pid]
    np.savez("points_normals.npz", p=pf, n=nf)
    if c is not None:
        cf = c[pid]
        print(f"Saving {pf.shape[0]} points...")
        pcu.save_mesh_vnc("pc_full.ply", pf, nf, cf)
        if len(cmd_args.spsr_path.strip()) > 0:
            os.system(f"{cmd_args.spsr_path} --in pc_full.ply --out recon.ply "
                      f"--width 0.1 --density --samplesPerNode 1.0")
    else:
        print(f"Saving {pf.shape[0]} points...")
        pcu.save_mesh_vn("pc_full.ply", pf, nf)
        if len(cmd_args.spsr_path.strip()) > 0:
            os.system(f"{cmd_args.spsr_path} --in pc_full.ply --out recon.ply "
                      f"--width 0.1 --density --samplesPerNode 1.0 --colors")
    print("Done!")

    
if __name__ == "__main__":
    main()
