import os
import h5py
import tifffile
import argparse
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="./data/train")
    parser.add_argument("--fragment_idx", type=str, default="1")
    params = parser.parse_args()

    fragment_path = os.path.join(params.datapath, params.fragment_idx)
    os.chdir(fragment_path)

    # Create a new HDF5 file
    hdf5_file = h5py.File('fragment.h5', 'w')

    # Loop through the TIFF slices and write each one to the HDF5 file
    for i in tqdm(range(65)):
        data = tifffile.imread(f'surface_volume/{str(i).zfill(2)}.tif')
        dataset_name = f'{i}'
        hdf5_file.create_dataset(dataset_name, data=data)

    # Close the files
    hdf5_file.close()