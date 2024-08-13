import os
import numpy as np
np.set_printoptions(suppress=True, precision=3)
import pickle
import h5py
import matplotlib.pyplot as plt
import wandb
from scipy.spatial.transform import Rotation as R

def print_dict_keys(dictionary, indent=0):
    for key, value in dictionary.items():
        print(" " * indent + str(key))
        if isinstance(value, dict):
            print_dict_keys(value, indent + 4)

# path = 'output_assisted/succ'
# for f_name in os.listdir(path):
#     if f_name.endswith('.pickle') or f_name == 'noisy_1':
#         continue
#     with open(f'{path}/{f_name}/traj_info.pickle', 'rb') as f:
#         traj_info = pickle.load(f)
#     print(traj_info['obj_in_hand']) 

# with open(f'{path}/teleop_traj_2.pickle', 'rb') as f:
#     traj = pickle.load(f)
#     print("traj: ", len(traj))
            
# org_pos, org_quat = np.array([ 0.50431009, -0.25087801,  0.46123985]), np.array([ 0.57241185,  0.58268626, -0.41505368,  0.4006892 ])
# org_matrix = R.from_quat(org_quat).as_matrix()
# print("org_matrix: ", org_matrix)
# T_r_g = np.vstack((np.column_stack((org_matrix, org_pos)), [0, 0, 0, 1]))
# print("T_r_g: ", T_r_g)


f = h5py.File('dynamics_model_dataset_seg_test/dataset.hdf5', "r")
print(np.array(f['data/episode_00007/actions/actions']))
img = np.array(f['data/episode_00007/observations/gripper_obj_seg'])
plt.imshow(img[0])
plt.show()
# f2 = h5py.File('dynamics_model_dataset_seg_test/dataset.hdf5', "r")
# print(np.array(f2['data/episode_00002/observations'].keys()))

# for i in range(16):
#     img = np.array(f[f'data/episode_{i+600:05}/observations/gripper_obj_seg'])
#     plt.imshow(img[0])
#     plt.show()

# img1 = np.array(f['data/episode_00394/observations/seg_instance_id'])
# img2 = np.array(f2['data/episode_00002/observations/seg_instance_id'])
# print("img1: ", img1.shape, img1.dtype)
# print("img2: ", img2.shape, img2.dtype)
# print("img1: ", img1[0])
# print("---------------------------")
# print("img2: ", img2[0])
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(img1[0])
# ax[1].imshow(img2[0])
# plt.show()


# succ_episodes = 0
# for k in f['data'].keys():
#     grasps = np.array(f['data'][k]['extras']['grasps'])
#     # print("len(grasps): ", grasps.shape)
#     # print("k, grasps: ", k, grasps)
#     # print("----------------")
#     if any(grasps):
#         succ_episodes += 1

# print("succ episodes: ", succ_episodes)

# plt.imshow(f['data/episode_00251/observations/gripper_obj_seg'][0])
# plt.show()
# for k in f['data'].keys():
#     print(k)
#     img = f[f'data/{k}/observations/gripper_obj_seg'][0]
#     h, w = img.shape[0], img.shape[1]

    # Create an empty array for the one-hot encoding
    # print("one_hot_encoded_image: ", one_hot_encoded_image.shape)

    # # Flatten the first two dimensions to use advanced indexing
    # flat_indices = img.reshape(-1) - 1  # -1 to convert label range [1-20] to [0-19]
    # one_hot_encoded_image.reshape(-1, 20)[np.arange(h*w), flat_indices] = 1
    # print("one_hot_encoded_image: ", one_hot_encoded_image.shape)
    
    # # Convert to one-hot encoding
    # h, w = img.shape[0], img.shape[1]
    # one_hot_encoded_image = np.zeros((h, w, 20), dtype=int)
    # for i in range(h):
    #     for j in range(w):
    #         label = img[i, j]
    #         one_hot_encoded_image[i, j, label] = 1 

    # print("one_hot_encoded_image: ", one_hot_encoded_image.shape)
    # print("img: ", img[50:60, 50:60])
    # print("one_hot_encoded_image: ", one_hot_encoded_image[50:60, 50:60])

    # img2 = np.zeros((h, w), dtype=int)
    # for i in range(h):
    #     for j in range(w):
    #         # print("np.where(one_hot_encoded_image[i][j] == 1): ", np.where(one_hot_encoded_image[i][j] == 1))
    #         img2[i][j] = np.where(one_hot_encoded_image[i][j] == 1)[0]

    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(img)
    # ax[1].imshow(img2)
    # plt.show()
        

#     print("img.shape: ", img.shape)
#     img_info = np.array(f[f'data/{k}/observations_info']['seg_instance_id']).astype(str)[0]
#     parts_of_concern = [  
#         '/World/robot0/gripper_right_link/visuals',
#         '/World/robot0/gripper_right_right_finger_link/visuals',
#         '/World/robot0/gripper_right_left_finger_link/visuals',
#         '/World/coffee_table_fqluyq_0/base_link/visuals',
#         '/World/box/base_link/visuals'
#     ]
#     ids_of_concern = []
#     for val in img_info:
#         print("val: ", val)
#         if val[1] in parts_of_concern:
#             ids_of_concern.append(int(val[0]))
    
#     print("ids_of_concern: ", ids_of_concern)
#     new_img = img.copy()
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             # print("img[i][j]: ", img[i][j], type(int(img[i][j])), type(ids_of_concern[0]))
#             if int(img[i][j]) not in ids_of_concern:
#                 # print(int(img[i][j]))
#                 new_img[i][j] = 0

    # # print(img_info)
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(img)
    # ax[1].imshow(new_img)
    # plt.show()

# # print(f.keys())
# # print("mask: ", np.array(f['mask']['valid']).shape)
# demos = f.keys()
# print("f.keys(): ", len(f.keys()))
# actions = np.array(f['episode_00016']['actions']['actions'])
# for a in actions:
#     print(np.linalg.norm(a[:3]))
# print("[")
# for sublist in actions:
#     formatted_sublist = [f"{elem:.4f}" for elem in sublist]
#     print("    [" + ", ".join(formatted_sublist) + "],")
#     # print("    [" + ", ".join(map(str, sublist)) + "],")
# print("]")


# print(f['episode_00000'].keys())
# for k in f['episode_00001']['actions']:
#     print("k: ", f['episode_00001']['actions'])
# for k in f['episode_00000']['observations'].keys():
#     print("k: ", f['episode_00000']['observations'][k])
# for k in f['episode_00000']['proprioceptions'].keys():
#     print("k: ", f['episode_00000']['proprioceptions'][k])
# for k in f['episode_00000']['extras'].keys():
#     print("k: ", f['episode_00000']['extras'][k])

# rgbs = np.array(f['episode_00003']['observations']['rgb'])
# actions = np.array(f['episode_00003']['actions']['actions'])
# for i in range(len(rgbs)):
#     print("action: ", actions[i])
#     print("contacts: ", np.array(f['episode_00001']['extras']['contacts'])[i])
#     print("grasps: ", np.array(f['episode_00001']['extras']['grasps'])[i])
#     fig, ax = plt.subplots(1,2)
#     ax[0].imshow(rgbs[i])
#     if i < len(rgbs) - 1:
#         print("11")
#         ax[1].imshow(rgbs[i+1])
#     else:
#         print("22")
#         ax[1].imshow(rgbs[i])
#     plt.show()

# print("-----")
# print("rgb: ", f['episode_00000']['observations']['rgb'])
# print("step keys: ", f['episode_00000']['step_00000'].keys())
# print("step attrs: ", f['episode_00000']['step_00000'].attrs['contacts'])
# print("obs: ", f['episode_00000']['step_00000']['obs'].keys())
# print("proprio: ", f['episode_00000']['step_00000']['proprio'].keys())
# print("proprio robot_pos: ", np.array(f['episode_00000']['step_00000']['proprio']['robot_pos']))
# print("proprio base_pos: ", np.array(f['episode_00000']['step_00000']['proprio']['base_pos']))
# print("preprocessed_actions: ", np.array(f['episode_00000']['step_00000']['preprocessed_actions']).shape)
# print("processed_actions: ", np.array(f['episode_00000']['step_00000']['processed_actions']).shape)

# rgb = f['episode_00000']['step_00000']['obs']['rgb']
# plt.imshow(rgb)
# plt.show()

# # show on wandb        
# wandb.login()
# run = wandb.init(
#     project="VMPC",
#     notes="trial experiment"
# )
# table_1 = wandb.Table(columns=["f_name", "Video"])
# rows = []
# path = '/home/arpit/test_projects/vp2/outputs/2024-06-17/12-51-21/fitvid_predictions'
# for filename in os.listdir(path):
#     rows.append([filename, wandb.Video(f'{path}/{filename}', fps=30, format="mp4")])

# table_1 = wandb.Table(
#     columns=table_1.columns, data=rows
# )
# run.log({"Videos": table_1}) 
            

# import pkg_resources, os, time

# for package in pkg_resources.working_set:
#     print("%s: %s" % (package, time.ctime(os.path.getctime(package.location))))


