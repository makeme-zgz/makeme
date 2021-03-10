import argparse
from utils import io_util
from utils import renderer
import cv2
import numpy as np
import trimesh
import json
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def _depth_map_from_shape(shape_filename, image_idx, cam_param, depth_map_dir):
    K = cam_param['K']
    Rt = cam_param['Rt'][:3, :]
    h = cam_param['height']
    w = cam_param['width']
    depth_map, _ = renderer.render_cvcam(
        K, Rt, shape_filename, std_size=(h, w))

    # Write depth map to the provide path if needed.
    if not os.path.exists(depth_map_dir):
        os.makedirs(depth_map_dir)
    io_util.write_pfm(os.path.join(
        depth_map_dir, f'{image_idx}.pfm'), depth_map)


def read_camera_params(camera_filename):
    images_param = []
    with open(camera_filename, 'r') as f:
        params = json.load(f)  # read parameters
    image_num = len(params) // 9  # get image number
    for i in range(image_num):
        images_param.append({
            'K': np.array(params['%d_K' % i]),
            'Rt': np.array(params['%d_Rt' % i] + [[0.0, 0.0, 0.0, 1.0]]),
            'distortion': np.array(params['%d_distortion' % i], dtype=np.float32),
            'height': params['%d_height' % i],
            'width': params['%d_width' % i],
            'valid': params['%d_valid' % i]})
    return images_param


def write_camera_param(cam_param, depth_ranges, cam_dir, image_idx):
    if not os.path.exists(cam_dir):
        os.makedirs(cam_dir)

    cam_filename = os.path.join(cam_dir, f'{image_idx}.txt')
    Rt = cam_param['Rt']
    K = cam_param['K']
    with open(cam_filename, 'w') as f:
        f.write('extrinsic\n')
        for j in range(4):
            for k in range(4):
                f.write(str(Rt[j][k]) + ' ')
            f.write('\n')
        f.write('\nintrinsic\n')
        for j in range(3):
            for k in range(3):
                f.write(str(K[j][k]) + ' ')
            f.write('\n')
        f.write('\n%f %f %f %f\n' % (
            depth_ranges[0], depth_ranges[1], depth_ranges[2], depth_ranges[3]))


def compute_depth_range(cam_param, shape_mesh, depth_number=None, interval_scale=1.0):
    Rt = cam_param['Rt']
    K = cam_param['K']

    zs = []
    for vertex in shape_mesh.vertices:
        transformed = np.matmul(Rt, [vertex[0], vertex[1], vertex[2], 1.0])
        zs.append(transformed[2].item())
    zs_sorted = sorted(zs)

    # relaxed depth range
    depth_min = zs_sorted[int(len(zs) * .01)]
    depth_max = zs_sorted[int(len(zs) * .99)]

    # Determine depth number using inverse depth setting. See MVSNet s.m.
    if depth_number is None:
        R = Rt[0:3, 0:3]
        t = Rt[0:3, 3]
        p1 = [K[0, 2], K[1, 2], 1]
        p2 = [K[0, 2] + 1, K[1, 2], 1]
        P1 = np.matmul(np.linalg.inv(K), p1) * depth_min
        P1 = np.matmul(np.linalg.inv(R), (P1 - t))
        P2 = np.matmul(np.linalg.inv(K), p2) * depth_min
        P2 = np.matmul(np.linalg.inv(R), (P2 - t))
        depth_num = (1 / depth_min - 1 / depth_max) / \
            (1 / depth_min - 1 / (depth_min + np.linalg.norm(P2 - P1)))
    else:
        depth_num = depth_number
    depth_interval = (depth_max - depth_min) / (depth_num - 1) / interval_scale

    return [depth_min, depth_interval, depth_num, depth_max]


def _undistort_images(raw_image_dir, undist_image_dir, image_idx, cam_param):
    if not os.path.exists(undist_image_dir):
        os.makedirs(undist_image_dir)

    K = cam_param['K']
    distort = cam_param['distortion']

    filename = f'{image_idx}.jpg'
    raw_img_filename = os.path.join(raw_image_dir, filename)
    undist_img_filename = os.path.join(undist_image_dir, filename)
    raw_image = cv2.imread(raw_img_filename)
    undist_img = cv2.undistort(raw_image, K, distort)
    cv2.imwrite(undist_img_filename, undist_img)


def _create_view_pair(data_dir, view_pairs_dir, exp_name):
    # SfM with sparse feature matching using COLMAP.
    colmap_sparse_cmd = f"""
        colmap automatic_reconstructor
        --workspace_path {data_dir}
        --image_path {data_dir}/images
        --sparse 1
        --dense 0
        --use_gpu 0
    """
    colmap_sparse_cmd = ' '.join(colmap_sparse_cmd.strip().split())
    print(f'Running SfM with Colmap cmd: {colmap_sparse_cmd}')
    os.system(colmap_sparse_cmd)

    # Undistort SfM results.
    colmap_undistort_cmd = f"""
        colmap image_undistorter
        --image_path {data_dir}/images
        --input_path {data_dir}/sparse/0
        --output_path {data_dir}/dense
        --output_type COLMAP
    """
    colmap_undistort_cmd = ' '.join(colmap_undistort_cmd.strip().split())
    print(f'Undistorting SfM result from Colmap. CMD: {colmap_undistort_cmd}')
    os.system(colmap_undistort_cmd)

    # Compute view pairs using the SfM results.
    from data import colmap_to_view_pair as c2v
    view_pairs = c2v.compute_view_pair(os.path.join(data_dir, 'dense'))

    if not os.path.exists(view_pairs_dir):
        os.makedirs(view_pairs_dir)

    view_pairs_filename = os.path.join(view_pairs_dir, f'{exp_name}.json')
    with open(view_pairs_filename, 'w') as f:
        json.dump(view_pairs, f)


def _create_train_data(data_root, max_count=None):
    train_data = []
    images_dir = os.path.join(data_root, 'images')
    view_pair_dir = os.path.join(data_root, 'view_pairs')
    shapes_idx = os.listdir(images_dir)

    def save_train_data(root_dir, data):
        train_data_filename = os.path.join(root_dir, 'train_data.json')
        with open(train_data_filename, 'w') as f:
            json.dump(data, f)

    for shape_idx in shapes_idx:
        exp_names = os.listdir(os.path.join(images_dir, shape_idx))
        for exp_name in exp_names:
            # Load view pairs file.
            view_pair_filename = os.path.join(
                view_pair_dir, shape_idx, f'{exp_name}.json')
            with open(view_pair_filename, 'r') as f:
                pairs = json.load(f)
            images = os.listdir(os.path.join(images_dir, shape_idx, exp_name))

            def image_filename(idx):
                return os.path.join('images', shape_idx, exp_name, f'{idx}.jpg')

            def cam_filename(idx):
                return os.path.join('cameras', shape_idx, exp_name, f'{idx}.txt')

            for ref_idx in range(len(images)):
                if max_count is not None and max_count <= len(train_data):
                    save_train_data(data_root, train_data)
                    return

                gt_depth_map = os.path.join(
                    'depth_map', shape_idx, exp_name, f'{ref_idx}.pfm')
                ref_image = image_filename(ref_idx)
                ref_cam = cam_filename(ref_idx)
                srcs_idx = pairs[ref_idx]
                sample = {
                    'ref': ref_image,
                    'ref_cam': ref_cam,
                    'srcs': [image_filename(src_idx) for src_idx in srcs_idx],
                    'srcs_cam': [cam_filename(src_idx) for src_idx in srcs_idx],
                    'gt': gt_depth_map,
                }
                train_data.append(sample)

    if max_count is None:
        save_train_data(data_root, train_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert facescape to mvs data')
    parser.add_argument('--data_root', default=None,
                        help='Root dir of Facescape data.')
    parser.add_argument('--gen_depth', default=False,
                        help='Whether or not to generate depth map.')
    parser.add_argument('--gen_cam', default=False,
                        help='Whether or not to generate camera file.')
    parser.add_argument('--undist', default=False,
                        help='Whether or not to undistort images.')
    parser.add_argument('--gen_view_pair', default=False,
                        help='Whether or not to image pairs.')
    parser.add_argument('--gen_train_data', default=False, const=True, nargs='?',
                        help='Whether or not to create train data file.')
    parser.add_argument('--reorg_raw_img_data', default=False,
                        help='Whether or not to reorganize the raw image data.')
    parser.add_argument('--max_row', default=None, type=int,
                        help='Max training data row.')
    args = parser.parse_args()

    curr_path = args.data_root if not args.data_root is None else os.getcwd()
    shape_dir = os.path.join(curr_path, 'shapes')
    raw_data_dir = os.path.join(curr_path, 'raw_data')

    shapes_idx = os.listdir(raw_data_dir)
    for idx in shapes_idx:
        exps_name = os.listdir(f'{raw_data_dir}/{idx}/')
        for exp_name in exps_name:
            if args.reorg_raw_img_data:
                import glob
                print(f'Reorganizing raw image data: {idx}/{exp_name}')
                raw_image_dir = os.path.join(raw_data_dir, idx, exp_name)
                image_dir = os.path.join(raw_image_dir, 'images')
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                for f in glob.glob(f'{raw_image_dir}/*.jpg'):
                    os.system(f'mv {f} {image_dir}')

            print(f'Processing data: {idx}/{exp_name}')
            shape_filename = os.path.join(shape_dir, idx, f'{exp_name}.ply')

            shape_mesh = None
            if args.gen_cam:
                shape_mesh = trimesh.load(shape_filename)

            # Create view pair map.
            raw_exp_data_dir = os.path.join(raw_data_dir, idx, exp_name)
            if args.gen_view_pair:
                print(f'Creating view pairs for {idx} / {exp_name}')
                view_pair_dir = os.path.join(curr_path, 'view_pairs', idx)
                _create_view_pair(raw_exp_data_dir, view_pair_dir, exp_name)

            if args.undist or args.gen_depth or args.gen_cam:
                cam_params = read_camera_params(
                    os.path.join(raw_exp_data_dir, 'params.json'))
                for image_idx, cam_param in enumerate(cam_params):
                    print(f'Processing {image_idx + 1} / {len(cam_params)}')

                    # Undistort images given camera parameters.
                    if args.undist:
                        undist_exp_image_dir = os.path.join(
                            curr_path, 'images', idx, exp_name)
                        _undistort_images(
                            os.path.join(raw_exp_data_dir, 'images'),
                            undist_exp_image_dir,
                            image_idx,
                            cam_param)

                    # Generate and write depth map from shape.
                    if args.gen_depth:
                        depth_map_dir = os.path.join(
                            curr_path, 'depth_map', idx, exp_name)
                        _depth_map_from_shape(
                            shape_filename, image_idx, cam_param, depth_map_dir)

                    # Convert camera parameters for MVS.
                    if args.gen_cam:
                        depth_ranges = compute_depth_range(
                            cam_param, shape_mesh)
                        mvs_cam_dir = os.path.join(
                            curr_path, 'cameras', idx, exp_name)
                        write_camera_param(
                            cam_param, depth_ranges, mvs_cam_dir, image_idx)
            break

    if args.gen_train_data:
        _create_train_data(curr_path, max_count=args.max_row)
