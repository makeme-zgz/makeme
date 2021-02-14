import json, sys, os, argparse
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

sys.path.append("../../utils")
import renderer
import io_util


def _depth_map_from_shape(shape_filename, cam_params, depth_map_dir, write=False):
    for i, param in enumerate(cam_params):
        print(f'Rendering: {i + 1} / {len(cam_params)}')
        K = param['K']
        Rt = param['Rt']
        h = param['height']
        w = param['width']
        depth_map, _ = renderer.render_cvcam(K, Rt, shape_filename, std_size=(h, w))

        # Write depth map to the provide path if needed.
        if write:
            if not os.path.exists(depth_map_dir):
                os.makedirs(depth_map_dir)
            io_util.write_pfm(os.path.join(depth_map_dir, f'{i}.pfm'), depth_map)


def read_camera_params(camera_filename):
    images_param = []
    with open(camera_filename, 'r') as f:
        params = json.load(f)  # read parameters
    image_num = len(params) // 9  # get image number
    for i in range(image_num):
        images_param.append({
            'K': params['%d_K' % i],
            'Rt': params['%d_Rt' % i],
            'distortion': params['%d_distortion' % i],
            'height': params['%d_height' % i],
            'width': params['%d_width' % i],
            'valid': params['%d_valid' % i]})
    return images_param


def generate_depth_map(shape_dir, camera_dir, depth_map_dir, write=False):
    if write and depth_map_dir is None:
        raise ValueError('Depth map directory should be provided if write is needed.')

    shapes_idx = os.listdir(camera_dir)
    for idx in shapes_idx:
        exps_name = os.listdir(f'{camera_dir}/{idx}/')
        for exp_name in exps_name:
            shape_filename = f'{shape_dir}/{idx}/{exp_name}.ply'
            camera_filename = f'{camera_dir}/{idx}/{exp_name}/params.json'
            depth_map_dir = f'{depth_map_dir}/{idx}/{exp_name}'
            cam_param = read_camera_params(camera_filename)

            # Generate and write depth map from shape.
            _depth_map_from_shape(shape_filename, cam_param, depth_map_dir, write=True)


def convert_to_mvs_data():
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert facescape to mvs data')
    parser.add_argument('--gen_depth', default=False, help='Whether or not to generate depth map.')
    args = parser.parse_args()

    curr_path = os.getcwd()
    shape_dir = os.path.join(curr_path, 'shapes')
    images_dir = os.path.join(curr_path, 'images')
    depth_map_dir = os.path.join(curr_path, 'depth_map')

    if args.gen_depth:
        generate_depth_map(shape_dir, images_dir, depth_map_dir)

    
