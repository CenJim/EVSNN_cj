import pandas as pd
import h5py
import numpy as np
import hdf5plugin

def chunk_2d_array(array, chunk_size):
    chunked_array = []
    for i in range(0, len(array[0]), chunk_size):
            chunked_array.append(array[:, i:i+chunk_size])
    return chunked_array

def get_dataset(file_path, width, height, num_events_per_pixel):
    # open HDF5 file
    with h5py.File(file_path, 'r') as f:
        # obtain the t x y p
        dataset = np.zeros(shape=(4, len(f['events/p'])))
        dataset[0] = f['events/t'][()]
        dataset[1] = f['events/x'][()]
        dataset[2] = f['events/y'][()]
        dataset[3] = f['events/p'][()]

        return chunk_2d_array(dataset, int(width*height*num_events_per_pixel))

if __name__ == '__main__':
    file_path = '../Zurich_City_Events_Left/events.h5'
    dataset = get_dataset(file_path, 640, 480, 0.5)
    # for event in dataset:
    #     print(event)
    #     break
    print(len(dataset))
    print(dataset[:3])
