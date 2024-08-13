import h5py
from filelock import FileLock
import numpy as np
import cv2
import os

def create_recursive_structure(group, depth, max_depth):
    if depth > max_depth:
        return

    # Create attributes for the group
    group.attrs['description'] = f'This is group at depth {depth}'
    group.attrs['depth'] = depth

    # Create a dataset within this group
    data = np.random.random((10, 10))
    dataset = group.create_dataset('dataset', data=data)
    dataset.attrs['description'] = f'This is a dataset at depth {depth}'
    dataset.attrs['shape'] = data.shape

    # Create a subgroup and recurse
    subgroup = group.create_group(f'subgroup_depth_{depth}')
    create_recursive_structure(subgroup, depth + 1, max_depth)

def create_hdf5_file(filename, max_depth):
    with h5py.File(filename, 'w') as f:
        root_group = f.create_group('root')
        create_recursive_structure(root_group, 1, max_depth)

def dump(hdf5_path, obs, log=False):
    with FileLock(hdf5_path + ".lock"):
        with h5py.File(hdf5_path, 'a') as f:
            print(len(f.keys()))
            group_idx = len(f.keys())
            root_group = f.create_group(f'{group_idx:05d}')
            max_depth = 3
            create_recursive_structure(root_group, 1, max_depth)

def save_video(images, save_folder):
    # Define the frame rate (number of frames per second)
    frame_rate = 10

    height, width, layers = images[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can change the codec to the one you prefer
    # fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
    output_video = f'{save_folder}/video.mp4'
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
    # print("output_video: ", output_video)
    # print("len(images): ", len(images))
    # Loop through the images and add them to the video
    for image in images:
        frame = image[:, :, :3]
        # plt.imshow(frame)
        # plt.show()
        video.write(frame)

    # Release the VideoWriter object
    video.release()

    # Optional: Close the OpenCV window opened during execution
    cv2.destroyAllWindows()