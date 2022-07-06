import os
import numpy as np
import point_cloud_utils as pcu
import tqdm
from src.common import load_pc_dat


def load_dat_rays(dat_file_path, split_arrays=True):

    lidar_dat = load_pc_dat(dat_file_path)

    if split_arrays:
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
    return pts.astype(np.float32), dirs.astype(np.float32), dist.astype(np.float32), clr.astype(np.float32)


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


def trim_surface(points, mesh_v, mesh_f, mesh_n, mesh_c, trim_distance):
    p = points
    v, f, n, c = mesh_v, mesh_f, mesh_n, mesh_c

    print("Trimming mesh...")
    print("Finding nearest neighbor")
    nn_dist, _ = pcu.k_nearest_neighbors(v.astype(np.float32), p.astype(np.float32), k=2)
    nn_dist = nn_dist[:, 1]
    print("Stacking faces")
    f_mask = np.stack([nn_dist[f[:, i]] < trim_distance for i in range(f.shape[1])], axis=-1)
    print("Filtering faces")
    f_mask = np.all(f_mask, axis=-1)
    f_trimmed = f[f_mask]

    return f_trimmed

def run_surface_reconstruction(lidar_dat_path, ouput_path, max_dist=60, start_at=0,stop_at=-1, ignore_color=True, mode='knn', radius=20, min_angle_thresh=90.0, trim_distance=0.225, spsr_path='external/PoissonRecon/PoissonRecon'):
    ''' Given a set of 3D lidar rays in space runs surface reconstruction using SPSR surface reconstruction method (https://hhoppe.com/poissonrecon.pdf)

    Args:
        lidar_dat_path (string): path to the lidar dat files
        output_path (string): output folder
        max_dist (float): Ignore fused points greater than this distance from the ego vehicle
        start_at (int): Start fusing input points at this index
        stop_at (int): Stop fusing input points after this index (-1 = use all frames up to the last)
        ignore_color (bool): Don't store color information in the reconstruction
        mode (string): How to collect neighbors to estimate normals. One of (knn, ball, ball-rbf)"
        radius (int/float): parameters used in normal estimation. Can be either radius or k in k-nn.
        min_angle_thresh (float): Drop points whose normal and view direction exceed this value (degrees)
        trim_distance (float): Trim faces from the reconstructed mesh which have a vertex whose nearest neighbor in the input point cloud exceeds this distance
        spsr_path (string): path to the SPSR executable

    ''' 
   
    if mode not in ['knn', 'ball', 'ball-rbf']:
        raise ValueError(f"Invalid --mode, must be one of 'knn', 'ball', 'ball-rbf' but got '{mode}'")

    # set up temp paths 
    temp_fused_path = os.path.join(ouput_path, 'full_pc.ply')
    temp_rec_path = os.path.join(ouput_path, 'rec_pc.ply')
    
    print("Loading points and ray directions...")
    p, d, c = load_fused_pc(lidar_dat_path,
                            max_dist=max_dist,
                            start_idx=start_at,
                            end_idx=stop_at,
                            ignore_color=ignore_color)

    print(f"Loaded {p.shape[0]} points.")
    p, d = p.astype(np.float32), d.astype(np.float32)

    print("Estimating normals...")
    if mode == 'knn':
        pid, nf = pcu.estimate_point_cloud_normals_knn(p, int(radius), view_directions=d,
                                                       drop_angle_threshold=np.deg2rad(min_angle_thresh))
    elif mode == 'ball':
        pid, nf = pcu.estimate_point_cloud_normals_ball(p, radius, view_directions=d,
                                                        weight_function="constant",
                                                        drop_angle_threshold=np.deg2rad(min_angle_thresh))
    elif mode == 'ball-rbf':
        pid, nf = pcu.estimate_point_cloud_normals_ball(p, radius, view_directions=d,
                                                        weight_function="rbf",
                                                        drop_angle_threshold=np.deg2rad(min_angle_thresh))

    pf = p[pid]

    if c is not None:
        cf = c[pid]
        print(f"Saving {pf.shape[0]} points...")
        pcu.save_mesh_vnc(temp_fused_path, pf, nf, cf)
        if len(spsr_path.strip()) > 0:
            os.system(f"{spsr_path} --in {temp_fused_path} --out {temp_rec_path} "
                      f"--width 0.1 --density --samplesPerNode 1.0")
    else:
        print(f"Saving {pf.shape[0]} points...")
        pcu.save_mesh_vn(temp_fused_path, pf, nf)
        if len(spsr_path.strip()) > 0:
            os.system(f"{spsr_path} --in {temp_fused_path} --out {temp_rec_path} "
                      f"--width 0.1 --density --samplesPerNode 1.0 --colors")
    print("Done!")

    v, f, n, c = pcu.load_mesh_vfnc(temp_rec_path)
    f_trimmed = trim_surface(pf, v, f, n, c, trim_distance)

    output_rec_path = os.path.join(ouput_path, 'reconstructed_mesh.ply')
    pcu.save_mesh_vfnc(output_rec_path, v, f_trimmed, n, c)

    # Clean up the temp files
    os.remove(temp_rec_path)
    os.remove(temp_fused_path)



if __name__ == "__main__":
    run_surface_reconstruction('/scratch/data/waymo_open/processed_data/10072231702153043603_5725_000_5745_000/lidar/','/scratch/data/waymo_open/processed_data/10072231702153043603_5725_000_5745_000/reconstruction/', max_dist=60)
