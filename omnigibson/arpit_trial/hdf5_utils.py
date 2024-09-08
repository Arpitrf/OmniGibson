import random
import h5py

import numpy as np
np.set_printoptions(suppress=True, precision=3)

def partition_dataset_train_valid(dataset):
    np.random.seed(0)
    random.seed(0)
    hdf5_file_path = dataset
    file = h5py.File(hdf5_file_path, 'a')
    all_keys = list(file['data'].keys())

    if 'mask' in file:
        print("deleting")
        input("Press enter to delete")
        del file['mask']
        
    print("file.keys: ", file.keys())

    # Shuffle the keys
    random.shuffle(all_keys)
    print("all_keys: ", all_keys)

    # Calculate the number of test samples (10% of the total data)
    num_test_samples = min(int(len(all_keys) * 0.1), 50)

    # Split the keys into train and test sets
    test_keys = all_keys[:num_test_samples]
    test_keys = [np.bytes_(s) for s in test_keys]
    train_keys = all_keys[num_test_samples:]
    train_keys = [np.bytes_(s) for s in train_keys]

    # Print the results
    print(f"Train keys: {len(train_keys)}")
    print(f"Test keys: {len(test_keys)}")
    
    file = h5py.File(hdf5_file_path, 'a')
    group_key = 'mask'
    if group_key not in file:
        group = file.create_group(group_key)
    group = file[group_key]

    group.create_dataset('train', data=train_keys, compression='gzip', compression_opts=9)
    group.create_dataset('valid', data=test_keys, compression='gzip', compression_opts=9)


# Utilitiy functions for copying / editing HDF5 files
        
def copy_attrs(src, dst):
    """ Copy attributes from one HDF5 object to another """
    for key, value in src.attrs.items():
        dst.attrs[key] = value

def copy_group(source_group, dest_group):
    for key in source_group:
        item = source_group[key]
        # print("key, type(item): ", key, type(item))
        if isinstance(item, h5py.Group):
            new_group = dest_group.create_group(key)
            copy_attrs(item, new_group)
            copy_group(item, new_group)

        elif isinstance(item, h5py.Dataset):
            dest_group.create_dataset(key, data=item[()], compression='gzip', compression_opts=9)
            copy_attrs(item, dest_group)

def edit_and_merge_hdf5(old_hdf5, new_hdf5):
    new_file = h5py.File(new_hdf5, 'a')
    group_key = 'data'
    if group_key not in new_file:
        group = new_file.create_group(group_key)
    group = new_file[group_key]

    old_file = h5py.File(old_hdf5, 'r')
    if 'data' in old_file.keys():
        old_file = old_file['data']
    copy_group(old_file, group) 

def reame_ep_and_merge_hdf5(old_hdf5, new_hdf5):
    new_file = h5py.File(new_hdf5, 'a')
    group_key = 'data'
    if group_key not in new_file:
        group = new_file.create_group(group_key)
    new_file = new_file[group_key]

    old_file = h5py.File(old_hdf5, 'r')
    if 'data' in old_file.keys():
        old_file = old_file['data']

    # find the last episode name in the destination hdf5 file
    counter = len(new_file)
    # print("counter: ", counter)

    for i, org_ep_name in enumerate(old_file):
        # New episode name in the destination hdf5 file
        new_ep_name = f'episode_{counter:05d}'
        # print("i, new_ep_name: ", i, new_ep_name)
        # Create a new group with new episode name in the destination hdf5 file
        new_group = new_file.create_group(new_ep_name)
        copy_group(old_file[org_ep_name], new_group) 
        counter += 1


def modify_dataset_in_hdf5_file(file):
    for i, ep in enumerate(file['data'].keys()):
        # Followed: https://stackoverflow.com/questions/22922584/how-to-overwrite-array-inside-h5-file-using-h5py
        actions = file[f'data/{ep}/actions/actions']
        del file[f'data/{ep}/actions/actions']
        seq_len = actions.shape[0]
        zeros = np.zeros((seq_len, 3))
        actions = np.hstack((zeros, actions))
        file.create_dataset(f'data/{ep}/actions/actions', data=actions)

def main():
    hdf5_file = '/home/arpit/test_projects/OmniGibson/pick_data/dataset.hdf5'
    file = h5py.File(hdf5_file, 'a')
    # modify_dataset_in_hdf5_file(file)
    # reame_ep_and_merge_hdf5('/home/arpit/test_projects/OmniGibson/navigation_data/dataset.hdf5', '/home/arpit/test_projects/OmniGibson/combined_data/dataset.hdf5')
    partition_dataset_train_valid('/home/arpit/test_projects/OmniGibson/combined_data/dataset.hdf5')

if __name__ == "__main__":
    main()