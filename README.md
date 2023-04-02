# scrollloader

``
pip install -r requirements.txt
``

Create HDF5 dataset from fragment .tiff files. For example,

``
python create_dataset.py --datapath './data/train' --fragment_idx 1
``

Then you can use the loader based on pytorch Dataset and DataLoader classes:

``
python loader.py --epoch_size 100000 --batch_size 64 --num_workers 16 --shape_x 96 --shape_y 96
``