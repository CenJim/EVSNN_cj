import pandas as pd
import h5py
import numpy as np
import hdf5plugin

def chunk_2d_array(array, chunk_size):
    chunked_array = []
    for i in range(0, len(array), chunk_size):
            chunked_array.append(array[i:i+chunk_size, :])
    return chunked_array

def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str, rectify_ev_maps):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

def get_dataset(file_path, width, height, num_events_per_pixel):
    # open HDF5 file
    with h5py.File(file_path + 'events.h5', 'r') as f:
        # obtain the t x y p
        dataset = np.zeros(shape=(len(f['events/p']), 4))
        dataset[:, 0] = f['events/t'][()]
        dataset[:, 1] = f['events/x'][()]
        dataset[:, 2] = f['events/y'][()]
        dataset[:, 3] = f['events/p'][()]
        
        with h5py.File(file_path + 'rectify_map.h5', 'r') as rec:
            rectify_map = rec['rectify_map'][()]
            xy_rect = rectify_map[dataset[:, 2].astype(int), dataset[:, 1].astype(int)]
            dataset[:, 1] = xy_rect[:, 0]   # rectify the x of event sequence
            dataset[:, 2] = xy_rect[:, 1]   # rectify the y of event sequence

        return chunk_2d_array(dataset, int(width*height*num_events_per_pixel))

if __name__ == '__main__':
    file_path = '../Zurich_City_Events_Left/'
    dataset = get_dataset(file_path, 640, 480, 0.5)
    # for event in dataset:
    #     print(event)
    #     break
    print(len(dataset))
    print(dataset[:3])
