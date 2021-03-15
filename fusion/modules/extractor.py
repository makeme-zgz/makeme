import torch
from torch import nn
from torch.nn.functional import normalize

from modules.utils import extract_values, get_index_mask, interpolate_indices


class Extractor(nn.Module):
    '''
        This module extracts voxel rays or blocks around voxels from the given volume state.
    '''

    def __init__(self):

        super(Extractor, self).__init__()


    def forward(
        self, 
        depth, 
        extrinsics, 
        intrinsics, 
        global_volume, 
        origin, 
        resolution, 
    ):
        '''
            Computes the forward pass of extracting the rays/blocks and the corresponding coordinates
            :param depth: depth map with the values that define the center voxel of the ray/block
            :param extrinsics: camera extrinsics matrix for mapping
            :param intrinsics: camera intrinsics matrix for mapping
            :param global_volume: voxel volume to extract values from
            :param origin: origin of voxel volume in world coordinates
            :param resolution: resolution of voxel volume
            :return: voxels of the given volume state as well as its coordinates and indices
        '''

        volume = global_volume

        intrinsics = intrinsics.float()
        extrinsics = extrinsics.float()

        device = depth.get_device()
        if device >= 0:
            intrinsics = intrinsics.to(device)
            extrinsics = extrinsics.to(device)
            volume = volume.to(device)

        # Step 1: Compute the world coordinates for pixels on the depth map.
        world_coords = self._compute_world_coordinates(depth, extrinsics, intrinsics)

        # Step 2: Compute rays from camera center to pixels on depth map.
        eye_w = extrinsics[:, :3, 3]
        ray_points, ray_direction = self._get_ray_points(
            world_coords, 
            eye_w,
            origin, 
            resolution, 
            n_points=int((self.n_points - 1)/2)
        )

        # Step 3: Interpolate the values for the sampled voxels using Nearest Neighbor Interpolation.
        interpolated_values, indices = self._interpolate(ray_points, volume)

        # Step 4: Reshape and pack the interpolated values for output.
        n1, n2, n3 = interpolated_values.shape
        indices = indices.view(n1, n2, n3, 8, 3)

        values = dict(
            interpolated_values=interpolated_values,
            ray_points=ray_points,
            ray_direction=ray_direction,
            indices=indices
        )

        del extrinsics, intrinsics, origin, volume, world_coords, eye_w
        return values


    def _compute_world_coordinates(self, depth, extrinsics, intrinsics):

        b, h, w = depth.shape
        n_points = h*w

        # generate frame meshgrid
        xx, yy = torch.meshgrid([torch.arange(h, dtype=torch.float),
                                 torch.arange(w, dtype=torch.float)])

        if torch.cuda.is_available():
            xx = xx.cuda()
            yy = yy.cuda()

        # flatten grid coordinates and bring them to batch size
        xx = xx.contiguous().view(1, h*w, 1).repeat((b, 1, 1))
        yy = yy.contiguous().view(1, h*w, 1).repeat((b, 1, 1))
        zz = depth.contiguous().view(b, h*w, 1)

        # generate points in pixel space
        points_p = torch.cat((yy, xx, zz), dim=2).clone() # b-(h*w) 3

        # invert
        intrinsics_inv = intrinsics.inverse().float()

        homogenuous = torch.ones((b, 1, n_points))

        if torch.cuda.is_available():
            homogenuous = homogenuous.cuda()

        # transform points from pixel space to camera space to world space (p->c->w)
        points_p[:, :, 0] *= zz[:, :, 0]
        points_p[:, :, 1] *= zz[:, :, 0]
        points_c = torch.matmul(intrinsics_inv, torch.transpose(points_p, dim0=1, dim1=2))
        points_c = torch.cat((points_c, homogenuous), dim=1)
        points_w = torch.matmul(extrinsics[:3], points_c)
        points_w = torch.transpose(points_w, dim0=1, dim1=2)[:, :, :3]

        del xx, yy, homogenuous, points_p, points_c, intrinsics_inv
        return points_w # b-(h*w)-3


    def _get_ray_points(self, coords, eye, origin, resolution, bin_size=1.0, n_points=4):

        direction = coords - eye # b-(h*w)-3
        direction = normalize(direction, p=2, dim=2)

        center_v = coords - origin
        points = [center_v]

        for i in range(1, n_points+1):
            point = center_v + i * bin_size * direction
            pointN = center_v - i * bin_size * direction
            points.append(point.clone())
            points.insert(0, pointN.clone())

        points = torch.stack(points, dim=2) # b-(h*w)-9-3
        direction = torch.stack(direction, dim=2) # b-(h*w)-3

        return points, direction


    def _interpolate(self, points, volume, mode='nearest'):

        # Get interpolation indices.
        def dist_func(wx, wy, wz):
            return wx * wy * wz

        indices, dists = interpolate_indices(points, dist_func)

        n1, n2, n3 = indices.shape 
        indices = indices.contiguous().view(n1*n2, n3).long() # (b*h*n*8)-3

        # Get valid indices
        valid = get_index_mask(indices, volume.shape)
        valid_idx = torch.nonzero(valid)[:, 0]

        vaild_values = extract_values(indices, volume, valid)
        value_container = torch.zeros_like(indices).double()
        value_container[valid_idx] = vaild_values
        value_container = value_container.view(dists.shape)

        # Interpolate based on the given mode.
        if mode == 'nearest':
            fusion_values = value_container[torch.min(dists).indices]
        elif mode == 'trilinear':
            fusion_values = torch.sum(value_container * dists, dim=1)
        
        # Reshape fusion values for output.
        b, h, n, _ = points.shape
        fusion_values = fusion_values.view(b, h, n)

        return fusion_values.float(), indices.view(n1, n2, n3)

