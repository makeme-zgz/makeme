import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import trimesh
import pyrender

def depth_map_from_shape(shape_filename, res_height, res_width, write=False, depth_map_filename=None):
    if write and depth_map_filename is None:
        raise ValueError('Depth map filename should be provide if write is needed.')

    shape_trimesh = trimesh.load(shape_filename)
    shape_mesh = pyrender.Mesh.from_trimesh(shape_trimesh)
    scene = pyrender.Scene()
    scene.add(shape_mesh)

    render = pyrender.OffscreenRenderer(viewport_width=res_width, viewport_height=res_height, point_size=1.0)
    _, depth_map = render.render(scene)

    # Write depth map to the provide path if needed.
    # if write:
    #     write_depth_map(depth, depth_map_filename)

    return depth_map


if __name__ == '__main__':
    

