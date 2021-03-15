import torch


def extract_values(indices, volume, mask=None):

    if mask is not None:
        x = torch.masked_select(indices[:, 0], mask)
        y = torch.masked_select(indices[:, 1], mask)
        z = torch.masked_select(indices[:, 2], mask)
    else:
        x = indices[:, 0]
        y = indices[:, 1]
        z = indices[:, 2]

    return volume[x, y, z]


def get_index_mask(indices, shape):

    xs, ys, zs = shape

    valid = ((indices[:, 0] >= 0) &
            (indices[:, 0] < xs) &
            (indices[:, 1] >= 0) &
            (indices[:, 1] < ys) &
            (indices[:, 2] >= 0) &
            (indices[:, 2] < zs))

    return valid

def interpolate_indices(points, dist_func):

    assert dist_func is not None

    # Origin voxel index.
    origin = torch.floor(points)

    alpha = torch.abs(points - origin)
    alpha_inv = 1 - alpha

    dists = []
    indices = []

    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                if i == 0:
                    dx = alpha[:, 0]
                    ix = origin[:, 0]
                else:
                    dx = alpha_inv[:, 0]
                    ix = origin[:, 0] + 1
                if j == 0:
                    dy = alpha[:, 1]
                    iy = origin[:, 1]
                else:
                    dy = alpha_inv[:, 1]
                    iy = origin[:, 1] + 1
                if k == 0:
                    dz = alpha[:, 2]
                    iz = origin[:, 2]
                else:
                    dz = alpha_inv[:, 2]
                    iz = origin[:, 2] + 1

                dists.append((dist_func(dx, dy, dz)).unsqueeze_(1))
                indices.append(torch.cat((ix.unsqueeze_(1),
                                          iy.unsqueeze_(1),
                                          iz.unsqueeze_(1)),
                                         dim=1).unsqueeze_(1))

    dists = torch.cat(dists, dim=1).unsqueeze_(-1) # (b*h*n)-8-1
    indices = torch.cat(indices, dim=1) # (b*h*n)-8-3

    del points, origin, alpha, alpha_inv, ix, iy, iz, dx, dy, dz

    return indices, dists


if __name__ == '__main__':
    points = torch.FloatTensor([[1.2, 1.2, 1.2]])

    def dist_func(dx, dy, dz):
        return dx + dy + dz
    
    indices, dists = interpolate_indices(points, dist_func)

