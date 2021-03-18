import torch
from torch import nn
from torch.nn.functional import normalize

from modules.utils import extract_values, get_index_mask, interpolate_indices


class Extractor(nn.Module):
    '''
        This module extracts voxel rays or blocks around voxels from the given volume state.
    '''

    def __init__(self, device, sample_number):

        super(Extractor, self).__init__()

        self._device = device
        self._sample_num = sample_number


    def forward(
        self, 
        depth_map, 
        extrinsics, 
        intrinsics, 
        global_volume, 
        volume_origin, 
    ):
        '''
            Computes the forward pass of extracting the rays/blocks and the corresponding coordinates
            :param depth_map: depth map with the values that define the center voxel of the ray/block
            :param extrinsics: camera extrinsics matrix for mapping
            :param intrinsics: camera intrinsics matrix for mapping
            :param global_volume: voxel volume to extract values from
            :param volume_origin: origin of voxel volume in world coordinates
            :return: voxels of the given volume state as well as its coordinates and indices
        '''

        volume = global_volume
        intrinsics = intrinsics.float()
        extrinsics = extrinsics.float()

        # Step 1: Compute the world coordinates for pixels on the depth map.
        world_coords = self._compute_world_coordinates(
            depth_map, 
            extrinsics, 
            intrinsics
        ) # b-(h*w)-3

        # Step 2: Compute rays from camera center to pixels on depth map.
        eye_w = extrinsics[:, :3, 3]
        ray_points, ray_directions = self._get_ray_points(
            world_coords, 
            eye_w,
            volume_origin, 
            self._sample_num
        ) # b-(h*w*n)-3, b-(h*w*n)-3

        # Step 3: Interpolate the values for the sampled voxels using Nearest Neighbor Interpolation.
        interpolated_values, indices = self._interpolate(
            ray_points, 
            volume,
        ) # b-(h*w*n)-1, b-(h*w*n)-3

        # Step 4: Finalize extraction output.
        _, h, w = depth_map.shape
        b, pts_per_batch, val_dim = interpolated_values.shape
        _, _, index_dim = indices.shape

        # Verify extraction result dimensions.
        assert pts_per_batch == h * w * self._sample_num
        assert val_dim == 1
        assert index_dim == 3
        
        # Reshape and pack the interpolated values for output.
        values = dict(
            interpolated_volume=interpolated_values.view(b, h, w, self._sample_num), # b-h-w-n
            indices=indices.view(b, h, w, self._sample_num, 3), # b-h-w-n-3
            ray_points=ray_points.view(b, h, w, self._sample_num, 3), # b-h-w-n-3
            ray_directions=ray_directions.view(b, h, w, self._sample_num, 3)[:, :, :, 0, :] # b-h-w-3
        )

        del extrinsics, intrinsics, volume_origin, volume, world_coords, eye_w
        return values


    def _compute_world_coordinates(self, depth_map, extrinsics, intrinsics):

        b, h, w = depth_map.shape
        n_points = h*w

        # generate frame meshgrid
        xx, yy = torch.meshgrid([torch.arange(h, dtype=torch.float),
                                 torch.arange(w, dtype=torch.float)])

        # flatten grid coordinates and bring them to batch size
        xx = xx.to(self._device).contiguous().view(1, h*w, 1).repeat((b, 1, 1))
        yy = yy.to(self._device).contiguous().view(1, h*w, 1).repeat((b, 1, 1))
        zz = depth_map.contiguous().view(b, h*w, 1)

        # generate points in pixel space
        points_p = torch.cat((yy, xx, zz), dim=2).clone() # b-(h*w)-3

        # invert
        intrinsics_inv = intrinsics.inverse().float()

        homogenuous = torch.ones((b, 1, n_points), device=self._device)

        # transform points from pixel space to camera space to world space (p->c->w)
        points_p[:, :, 0] *= zz[:, :, 0]
        points_p[:, :, 1] *= zz[:, :, 0]
        points_c = torch.matmul(intrinsics_inv, torch.transpose(points_p, dim0=1, dim1=2))
        points_c = torch.cat((points_c, homogenuous), dim=1)
        points_w = torch.matmul(extrinsics[:3], points_c)
        points_w = torch.transpose(points_w, dim0=1, dim1=2)[:, :, :3]

        del xx, yy, homogenuous, points_p, points_c, intrinsics_inv
        return points_w # b-(h*w)-3


    def _get_ray_points(self, coords, eye, volume_origin, n_points, step_size=1.0):

        assert n_points >= 1

        _, pts_per_batch, _ = coords.shape
        coords = torch.repeat_interleave(coords, n_points, dim=1) # b-(h*w*n)-3

        directions = coords - eye # b-(h*w*n)-3
        directions = normalize(directions, p=2, dim=2)

        # Get the beginning samples.
        start_idx = int((n_points - 1)/2)
        volume_coords = coords - volume_origin 
        start_points = volume_coords - start_idx * step_size * directions

        # Sample points along each ray direction.
        idx_list = torch.arange(
            0, 
            n_points, 
            dtype=coords.dtype, 
            device=coords.device
        ).view(1, n_points, 1).repeat((1, pts_per_batch, 1))
        points = start_points + step_size * directions * idx_list # b-(h*w*n)-3

        del idx_list, volume_coords, start_points
        return points, directions


    def _interpolate(self, points, volume):
        '''
            Interpolate the values for input points from the given volume.
            :param points: points to be interpolated. Dim: b-(h*w*n)-3
            :param volume: voxel volume to interpolate values from. Dim: x-y-z
            :return: voxels of the given volume state as well as its coordinates and indices
        '''

        # Get interpolation indices.
        def dist_func(wx, wy, wz):
            return wx * wy * wz

        # Get interpolation indices and distances.
        batch, pts_per_batch, dim = points.shape
        points = points.contiguous().view(batch * pts_per_batch, dim)
        indices, dists = interpolate_indices(points, dist_func) # (b*(h*w)*n)-8-3, (b*(h*w)*n)-8-1

        pts_num, interplt_num, dim = indices.shape 
        interplt_pts_num = pts_num * interplt_num
        assert batch * pts_per_batch == pts_num

        indices = indices.contiguous().view(interplt_pts_num, dim).long() # (b*(h*w)*n*8)-3

        # Get valid indices
        valid = get_index_mask(indices, volume.shape)
        valid_idx = torch.nonzero(valid)[:, 0]

        valid_values = extract_values(indices, volume, valid)
        volume_values = torch.zeros(interplt_pts_num, device=device) # (b*(h*w)*n*8)
        volume_values[valid_idx] = valid_values
        volume_values = volume_values.view(dists.shape) # (b*(h*w)*n)-8-1

        # Interpolate based on the given mode.
        indices = indices.view(pts_num, interplt_num, dim) # (b*(h*w)*n)-8-3
        fusion_values, indices = self._nearest_neighbor_interpolate(
            volume_values, 
            indices, 
            dists
        ) # (b*(h*w)*n)-1, (b*h*w*n)-3

        # Note that dimension of the returned indices is depended on the mode of interpolation.
        fusion_values = fusion_values.view(batch, pts_per_batch, 1)
        indices = indices.view(batch, pts_per_batch, dim)

        del points, dists, valid_values, volume_values
        return fusion_values, indices


    def _nearest_neighbor_interpolate(self, values, indices, dists):
        pts_num, _, dim = indices.shape

        # Get indices of min values in the interpolated distances.
        min_value_indices = torch.squeeze(torch.min(dists, dim=1).indices)

        # Extract min values and its index from the volume values. 
        fusion_values = values[torch.arange(pts_num), min_value_indices, :]
        indices = indices[torch.arange(pts_num), min_value_indices, :]

        # Reshape and return interpolated values.
        indices = indices.view(pts_num, dim) # (b*h*w*n)-3

        del min_value_indices
        return fusion_values, indices


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extractor = Extractor(device, sample_number=3)

    depth_map = torch.FloatTensor([[[1.0, 2.0]]]).to(device)
    extrinsic = torch.eye(4).to(device).view(1, 4, 4)
    intrinsic = torch.eye(3).to(device).view(1, 3, 3)
    volume = torch.ones(4, 4, 4, device=device)
    origin = torch.FloatTensor([-0.4, -0.4, -0.4]).to(device)

    extrinsic[0, 3, 3] = 0.0
    volume[0, 0, 0] = 2.0
    volume[0, 0, 1] = 3.0
    volume[0, 0, 2] = 4.0
    volume[2, 0, 2] = 5.0
    volume[3, 0, 3] = 6.0

    # world_coords = extractor._compute_world_coordinates(depth_map, extrinsic, intrinsic)
    # eye = extrinsic[:, :3, 3]
    # pts, directions = extractor._get_ray_points(world_coords, eye, origin, 3)
    # fusion_values, indices = extractor._interpolate(pts, volume)

    values = extractor.forward(depth_map, extrinsic, intrinsic, volume, origin)

    result = 2.0 + 3.0 + 4.0 + 5.0 * 2 + 6.0
    assert torch.sum(values['interpolated_volume']) == result

    interpolated_volume = values['interpolated_volume']
    indices = values['indices']
    ray_pts = values['ray_points']
    ray_directions = values['ray_directions']

    assert interpolated_volume.shape == (1, 1, 2, 3)
    assert indices.shape == (1, 1, 2, 3, 3)
    assert ray_pts.shape == (1, 1, 2, 3, 3)
    assert ray_directions.shape == (1, 1, 2, 3)
