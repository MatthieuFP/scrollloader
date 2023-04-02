import argparse
import os
import random
import torch
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
import h5py


class HDF5SubVolume(data.Dataset):
    def __init__(self, params) -> None:
        super(HDF5SubVolume, self).__init__()

        self.params = params
        # Paths of the hdf5 fragments
        self.data_paths = {f"{i}": os.path.join("data", "train", f"{str(i)}", "fragment.h5")
                           for i in range(1, 4)}

        self.shape_x = params.shape_x
        self.shape_y = params.shape_y
        self.z_slices = (20, 48)

    def __len__(self):
        return self.params.epoch_size

    def __getitem__(self, item):
        fragment_idx = str(random.randint(1, 3))
        with h5py.File(self.data_paths[fragment_idx]) as data:
            max_x, max_y = data["0"].shape
            x = random.randint(self.shape_x // 2, max_x - self.shape_x)
            y = random.randint(self.shape_y // 2, max_y - self.shape_y)

            patches = []
            for i in range(self.z_slices[0], self.z_slices[1] + 1):
                slice = data[f'{i}']
                patches.append(slice[x - self.shape_x // 2 + 1: x + self.shape_x // 2 + 1,
                                          y - self.shape_y // 2 + 1: y + self.shape_y // 2 + 1])

        patches = torch.from_numpy(np.stack(patches) / 255)
        return patches


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--shape_x", type=int, default=256)
    parser.add_argument("--shape_y", type=int, default=256)
    params = parser.parse_args()

    dataset = HDF5SubVolume(params)
    loader = data.DataLoader(dataset, batch_size=params.batch_size, num_workers=params.num_workers,
                             shuffle=True, pin_memory=True)

    for idx, region_voxel in tqdm(enumerate(loader)):
        continue
