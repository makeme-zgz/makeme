import torch

from modules.utils import extract_values, get_index_mask

class Integrator(torch.nn.Module):
    '''
        This module integrates the updated view-aligned feature volume back
        into the global state by inversing the nearest neighbor interpolation
        operation.
    '''

    def __init__(self, device):

        super(Integrator, self).__init__()

        self._device = device

    def forward(self, values, weights, indices, values_volume, weights_volume):
        '''
            Computes the updated global volume by integrating the given view aligned values 
            using the corresponding volume coordinates.

            Note that in order to aggregate colliding updates using average pooling, a weight tensor
            consists of 1's should be passed as input.

            :param values: updated view aligned feature values. Dim: b-h-w-n
            :param weights: distance from the sampled points to interpolated coordinates. Dim: b-h-w-n
            :param indices: corresponding volume coordinates to be updated. Dim: b-h-w-n-3
            :param values_volume: current global feature volume. Dim: x-y-z
            :param weights_volume: current global weight volume. Dim: x-y-z
            :return: updated global feature and weight volume. Dim: x-y-z, x-y-z
        '''

        # Reshape and flatten tensors
        b, h, w, n = values.shape
        pts_num = b * h * w * n
        values = values.contiguous().view(pts_num, 1)
        weights = weights.contiguous().view(pts_num, 1)
        indices = indices.contiguous().view(pts_num, 3)

        # Retrieve valid indices for update.
        valid = get_index_mask(indices, values_volume.shape)
        values = torch.masked_select(values[:, 0], valid)
        weights = torch.masked_select(weights[:, 0], valid)
        indices = torch.cat((
            torch.masked_select(indices[:, 0], valid).unsqueeze_(1),
            torch.masked_select(indices[:, 1], valid).unsqueeze_(1),
            torch.masked_select(indices[:, 2], valid).unsqueeze_(1)
        ), dim=1)

        # Compute the flatten indices.
        xs, ys, zs = values_volume.shape
        index = ys * zs * indices[:, 0] + zs * indices[:, 1] + indices[:, 2]

        # Aggregate the colliding updates and weights. 
        wcache = torch.zeros(values_volume.shape, device=self._device).view(xs * ys * zs)
        vcache = torch.zeros(values_volume.shape, device=self._device).view(xs * ys * zs)

        wcache.index_add_(0, index, weights)
        vcache.index_add_(0, index, weights * values)

        wcache = wcache.view(xs, ys, zs)
        vcache = vcache.view(xs, ys, zs)

        update = extract_values(indices, vcache)
        weights = extract_values(indices, wcache)

        # Update the current volumes with the updated values.
        values_old = extract_values(indices, values_volume)
        weights_old = extract_values(indices, weights_volume)

        value_update = (weights_old * values_old + update) / (weights_old + weights)
        weight_update = weights_old + weights

        values_volume[indices[:, 0], indices[:, 1], indices[:, 2]] = value_update
        weights_volume[indices[:, 0], indices[:, 1], indices[:, 2]] = weight_update

        return values_volume, weights_volume

