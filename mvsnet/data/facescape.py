import json
import os
import sys
import copy
import tqdm

import cv2
import numpy as np
import torch.utils.data as data

from utils.preproc import image_net_center as center_image, to_channel_first, resize, center_crop, recursive_apply
from utils.io_utils import load_cam, load_pfm

from data.data_utils import dict_collate, Until


class Facescape(data.Dataset):

    def __init__(self, root, train_data_filename, num_src, read, transforms):
        self.root = root
        with open(os.path.join(root, train_data_filename)) as f:
            self.train_data = json.load(f)
        self.num_src = num_src
        self.read = read
        self.transforms = transforms

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, i):
        filenames = copy.deepcopy(self.train_data[i])
        recursive_apply(filenames, lambda fn: os.path.join(self.root, fn))

        sample = self.read(filenames)
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def read(filenames):
    ref_name, ref_cam_name, srcs_name, srcs_cam_name, gt_name = [filenames[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt']]
    ref, *srcs = [cv2.imread(fn) for fn in [ref_name] + srcs_name]
    ref_cam, *srcs_cam = [load_cam(fn) for fn in [ref_cam_name] + srcs_cam_name]
    gt = np.expand_dims(load_pfm(gt_name), -1)
    masks = [(np.ones_like(gt)*255).astype(np.uint8) for _ in range(len(srcs))]
    return {
        'ref': ref,
        'ref_cam': ref_cam,
        'srcs': srcs,
        'srcs_cam': srcs_cam,
        'gt': gt,
        'masks': masks,
        'skip': 0
    }


def train_preproc(sample, preproc_args):
    ref, ref_cam, srcs, srcs_cam, gt, masks, skip = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks', 'skip']]

    ref, *srcs = [center_image(img) for img in [ref] + srcs]
    ref, ref_cam, srcs, srcs_cam, gt, masks = resize([ref, ref_cam, srcs, srcs_cam, gt, masks], preproc_args['resize_width'], preproc_args['resize_height'])
    ref, *srcs, gt = to_channel_first([ref] + srcs + [gt])
    masks = to_channel_first(masks)

    srcs, srcs_cam, masks = [np.stack(arr_list, axis=0) for arr_list in [srcs, srcs_cam, masks]]

    return {
        'ref': ref,  # 3hw
        'ref_cam': ref_cam,  # 244
        'srcs': srcs,  # v3hw
        'srcs_cam': srcs_cam,  # v244
        'gt': gt,  # 1hw
        'masks': masks,  # v1hw
        'skip': skip  # scalar
    }


def get_train_loader(root, num_src, total_steps, batch_size, preproc_args, num_workers=0):
    dataset = Facescape(
        root, 'train_data.json', num_src,
        read=lambda filenames: read(filenames),
        transforms=[lambda sample: train_preproc(sample, preproc_args)]
    )
    loader = data.DataLoader(dataset, batch_size, collate_fn=dict_collate, shuffle=True, num_workers=num_workers, drop_last=True)
    cyclic_loader = Until(loader, total_steps)
    return dataset, cyclic_loader


if __name__ == '__main__':
    data_root = '/home/zgz/makeme/data/facescape'
    _, loader = get_train_loader(
        data_root, 3, 100, 2,
        {
            'resize_width': 640,
            'resize_height': 512,
        },
    )
    pbar = tqdm.tqdm(loader, dynamic_ncols=True)
    for sample in pbar:
        pass