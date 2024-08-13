import cv2
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pickle
import h5py

def save_video(path, f_name, images):
    imgio_kargs = {'fps': 2, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}

    output_path = f'{path}/{f_name}.mp4'
    writer = imageio.get_writer(output_path, **imgio_kargs)  
    for image in images:
        writer.append_data(image)
    writer.close()

def read_img_save_video(path, f_name, img_folder):
    imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    
    image_folder = f'{path}/{f_name}/{img_folder}'
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # Sort the images by their filename using numeric order
    # images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    images.sort(key=lambda x: int(x.split('.')[0]))

    output_path = f'{path}/{f_name}/video_{img_folder}.mp4'
    writer = imageio.get_writer(output_path, **imgio_kargs)  
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

        writer.append_data(frame)
    writer.close()
    # return output_path

def read_img_save_video2(path, f_name):
    # Define the frame rate (number of frames per second)
    frame_rate = 10

    # Get the list of image files in the directory
    image_folder = f'{path}/{f_name}/rgb'
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # Sort the images by their filename using numeric order
    # images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    images.sort(key=lambda x: int(x.split('.')[0]))

    # Get the first image to determine the frame size
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can change the codec to the one you prefer
    fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
    output_video = f'{path}/{f_name}/video.mp4'
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
    print("output_video: ", output_video)
    print("len(images): ", len(images))
    # Loop through the images and add them to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        # plt.imshow(frame)
        # plt.show()
        video.write(frame)

    # Release the VideoWriter object
    video.release()

    # Optional: Close the OpenCV window opened during execution
    cv2.destroyAllWindows()

def save_seg_video(path, f_name):
    # load original segmentation images
    obs_lis = []
    print("--", f'{path}/{f_name}/seg_display')
    os.makedirs(f'{path}/{f_name}/seg_display', exist_ok=True)

    for obs in os.listdir(f'{path}/{f_name}/obs'):
        obs_lis.append(obs)
    obs_lis.sort(key=lambda x: int(x.split('.')[0]))
    for obs in obs_lis:
        with open(f'{path}/{f_name}/obs/{obs}', 'rb') as f:
            obs_dict = pickle.load(f)
            seg = obs_dict['robot0']['robot0:eyes:Camera:0']['seg_semantic']

            # Get unique labels in the segmentation image
            # unique_labels = np.unique(seg)
            # print("unique_labels: ", unique_labels)
            mapping = {
                825831922: 1,
                2814990211: 2,
                2848186650: 3, 
                # 3157253163: 4,
                298104422: 4,
                4013582681: 5
            }
            for i in  range(seg.shape[0]):
                for j in range(seg.shape[1]):
                    seg[i][j] = mapping[seg[i][j]]

            plt.imshow(seg)
            file_name = obs.split('.')[0]
            plt.savefig(f'{path}/{f_name}/seg_display/{file_name}.jpg')
            plt.close()

    read_img_save_video(path, f_name, 'seg_display')
            

# path = 'output_assisted/goal_traj'
# for f_name in os.listdir(path):
#     # if f_name != '0':
#     #     continue
#     read_img_save_video(path, f_name, 'rgb')
#     save_seg_video(path, f_name)
    
# # save sequence of images as videos
# f = h5py.File('temp_dataset/dataset.hdf5', "r") 
# path = '/home/arpit/test_projects/OmniGibson/temp_dataset/videos'
# os.makedirs(path, exist_ok=True)
# counter = 0
# for k in f['data'].keys():
#     # if counter > 15:
#     #     break
#     rgbs = np.array(f['data'][k]['observations']['seg_instance_id'])
#     # rgbs = rgbs[:, :, :, :3]
#     save_video(path, k, rgbs)
#     counter += 1

# save sequence of images on disk
f = h5py.File('dynamics_model_dataset_seg_test/dataset.hdf5', "r") 
counter = 0
for k in f['data'].keys():
    path = f'/home/arpit/test_projects/OmniGibson/dynamics_model_dataset_seg_test/{k}_images'
    os.makedirs(path, exist_ok=True)
    if counter > 15:
        break
    gripper_obj_segs = np.array(f['data'][k]['observations']['gripper_obj_seg'])
    for i, img in enumerate(gripper_obj_segs):
        plt.imshow(img)
        plt.savefig(os.path.join(path , f'{i:04}.jpg'))
        # cv2.imwrite(os.path.join(path , f'{i:04}.jpg'), img)
    # rgbs = np.array(f['data'][k]['observations']['rgb'])
    # rgbs = rgbs[:, :, :, :3]
    # for i, bgr in enumerate(rgbs):
    #     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    #     cv2.imwrite(os.path.join(path , f'{i:04}.jpg'), rgb)
    counter += 1
    
# edit hdf5 files
