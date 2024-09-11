import os
import numpy as np
np.set_printoptions(suppress=True, precision=3)
import pickle
import h5py
import matplotlib.pyplot as plt
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

def extract_observations_info_from_hdf5(obs_info_strings, obs_info_shapes):
        # Reconstruct original structure
        idx = 0
        reconstructed_data = []
        for shape in obs_info_shapes:
            sublist = []
            for _ in range(shape):
                sublist.append(list(map(lambda x: x.decode('utf-8'), obs_info_strings[idx:idx+2])))
                idx += 2
            reconstructed_data.append(sublist)

        reconstructed_data = np.array(reconstructed_data, dtype=object)
        # for i in range(len(reconstructed_data)):
        #     print(i, np.array(reconstructed_data[i]).shape)
        #     if i == 0:
        #         print(reconstructed_data)
        return reconstructed_data

def get_seg_instance_id_info(f, ep):
    # Basically dealing with HDF5 limitation: handling inconsistent length arrays in observations_info/seg_instance_id
    if 'seg_instance_id_strings' in f[f'data/{ep}/observations_info'].keys():
        seg_instance_id_strings = np.array(f["data/{}/observations_info/seg_instance_id_strings".format(ep)])
        seg_instance_id_shapes = np.array(f["data/{}/observations_info/seg_instance_id_shapes".format(ep)])
        seg_instance_id = extract_observations_info_from_hdf5(obs_info_strings=seg_instance_id_strings, 
                                                                    obs_info_shapes=seg_instance_id_shapes)
    else:
        hd5key = "data/{}/observations_info/seg_instance_id".format(ep)
        seg_instance_id = np.array(f[hd5key]).astype(str)
    return seg_instance_id

def obtain_gripper_obj_seg(img, img_info):
        # img = f[f'data/{k}/observations/seg_instance_id'][0]
        # img_info = np.array(f[f'data/{k}/observations_info']['seg_instance_id']).astype(str)[0]
        
        # OPTION 2: Make fixed classes for each object (useful when need different weights for different classes in the cross entropy loss)
        obs_info_with_new_cls_padded = []
        img_copy = img.copy()
        obs_info = img_info
        old_to_new_class_dict = {i: i for i in range(20)}
        fixed_classes = {
            '/World/robot0/gripper_right_link/visuals': 1,
            '/World/robot0/gripper_right_right_finger_link/visuals': 1,
            '/World/robot0/gripper_right_left_finger_link/visuals': 1,
            '/World/coffee_table_fqluyq_0/base_link/visuals': 2,
            '/World/box/base_link/visuals': 3,
            '/World/mixing_bowl/base_link/visuals': 4 
        }
        # print("obs_info shape: ", obs_info.shape)
        for class_name, target_class_value in fixed_classes.items():
            found_class = False
            class_value_in_current_episode = -1
            # class_value_in_current_frame = obs_info[0, obs_info[:, 0] == class_name, 1]
            for seq_num in range(len(obs_info)):
                for i, cls in enumerate(obs_info[seq_num]):
                    print("cls[1], class_name: ", cls[1], class_name)
                    if cls[1] == class_name:
                        # print("cls[1], class_name: ", cls[1], class_name)
                        class_value_in_current_episode = int(cls[0])
                        found_class = True
                        break
                if found_class:
                    break

            if class_value_in_current_episode != -1:
                # print("class_name, current class value: ", class_name, class_value_in_current_episode)
                img_copy[img == class_value_in_current_episode] = target_class_value
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(img[0])
        # ax[1].imshow(img_copy[0])
        # plt.show()
        print("img_copy: ", img_copy.shape)
        return img_copy


f = h5py.File('combined_data/dataset.hdf5', "r")
# ======================== Basic hdf5 testing ======================
# f = h5py.File('prior/dataset.hdf5', "r")
print("len: ", len(f['data']))
# print(np.array(f['data/episode_00004/actions/actions']))
# print("--", f[f'data/episode_00001/observations_info'].keys())
# print("--", np.array(f[f'data/episode_00001/observations_info']['seg_instance_id_strings']))
# for ep in f['data'].keys():
#     print(ep, np.array(f['data'][f'{ep}/actions/actions']).shape)
#     print("ep: ", ep)
#     print("--", np.array(f[f'data/{ep}/observations_info']['seg_instance_id']))
# obs_info = f['data/episode_00000/observations_info'].keys()
# print("obs_info: ", obs_info)
# grasped_state = np.array(f['data/episode_00000/extras/grasps'])
# print("grasped_state: ", grasped_state)
# img = np.array(f[f'data/episode_00300/observations/gripper_obj_seg'])
# print("img.shape: ", img.shape)
# fig, ax = plt.subplots(1, img.shape[0])
# for i in range(len(img)):
#     ax[i].imshow(img[i])
# plt.show()
# ===================================================================

# # ======================= Episodes with grasps ======================
# succ_episodes = 0
# for k in f['data'].keys():
#     grasps = np.array(f['data'][k]['extras']['grasps'])
#     # print("grasp.shape: ", grasps.shape)
#     # print("k, grasps: ", k, grasps)
#     # print("----------------")
#     # if any(grasps):
#     if grasps[-1]:
#         print("k, grasps: ", k, grasps)
#         succ_episodes += 1
# print("succ episodes: ", succ_episodes)
# # ======================================================================


# plt.imshow(f[f'data/episode_00002/observations/gripper_obj_seg'][0])
# plt.show()

# episodes = ['episode_00000', 'episode_01000', 'episode_01300']
# fig, ax = plt.subplots(1,3)
# for i, ep in enumerate(episodes):
#     img = np.array(f[f'data/{ep}/observations/gripper_obj_seg'])
#     img_info = get_seg_instance_id_info(f, ep)
#     input()
#     # img_info = np.array(f[f'data/{ep}/observations_info']['seg_instance_id']).astype(str)
#     new_img = obtain_gripper_obj_seg(img, img_info)
#     # ax[i].imshow(np.array(f[f'data/{ep}/observations/gripper_obj_seg'][0]))
#     ax[i].imshow(new_img[0])
# plt.show()

# print(f['data/episode_00000/observations'].keys())
# rgb = np.array(f['data/episode_00280/observations/rgb'])
# print(rgb)
# for i in range(len(actions)):
#     print(actions[i], np.linalg.norm(actions[i,:2]))

# # obs = np.array(f['data/episode_00001/observations/seg_instance_id'][2])
# obs_info_strings = np.array(f['data/episode_00000/observations_info/seg_instance_id_strings'])
# print("obs_info_strings: ", obs_info_strings.shape)
# string_array = np.vectorize(lambda x: x.decode('utf-8'))(obs_info_strings)
# print("string_array: ", string_array)
# obs_info_shapes = np.array(f['data/episode_00000/observations_info/seg_instance_id_shapes'])
# print("obs_info_shapes: ", obs_info_shapes)

# # Reconstruct original structure
# idx = 0
# reconstructed_data = []
# for shape in obs_info_shapes:
#     sublist = []
#     for _ in range(shape):
#         sublist.append(list(map(lambda x: x.decode('utf-8'), obs_info_strings[idx:idx+2])))
#         idx += 2
#     reconstructed_data.append(sublist)

# reconstructed_data = np.array(reconstructed_data, dtype=object)
# for i in range(len(reconstructed_data)):
#     print(i, np.array(reconstructed_data[i]).shape)
#     if i == 0:
#         print(reconstructed_data)

# obs = np.array(f['data/episode_00000/observations/seg_instance_id'][0])
# obs_info = reconstructed_data[0]
# obtain_gripper_obj_seg(obs, obs_info)


# f = h5py.File('dynamics_model_dataset_seg_test/dataset.hdf5', "r")
# print(np.array(f['data/episode_00007/actions/actions']))
# img = np.array(f['data/episode_00007/observations/gripper_obj_seg'])
# plt.imshow(img[0])
# plt.show()
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

# # ================= show on wandb ===================        
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
# ======================================================
            



# ========= Testing out the output of get_prior from slahmr-hamer =============
# # out = np.load('/home/arpit/test_projects/slahmr-hamer/outputs_for_moma/pick_place_apple_navigate_prior_results.npz')
# out = np.load('/home/arpit/test_projects/slahmr-hamer/outputs_for_moma/forward_test3_prior_results.npz')
# print("out.shape: ", out.files)
# init_pos = out['hand_positions'][0]
# for pos in out['hand_positions']:
#     print("pos, norm: ", pos, np.linalg.norm(pos-init_pos))
#     final_pos =  out['hand_positions'][-1]
# print("final_pos - init_pos: ", final_pos - init_pos)
    

# out = np.load('/home/arpit/test_projects/slahmr-hamer/outputs_for_moma/nav_test6_prior_results.npz')
# init_pos = out['body_positions'][0]
# for pos in out['body_positions']:
#     print("pos, norm: ", pos, np.linalg.norm(pos-init_pos))

# final_pos =  out['body_positions'][-1]
# print("final_pos - init_pos: ", final_pos - init_pos)
# =============================================================================


# # ================ checking sampling ======================
# mu_z = [-0.073, -0.056, -0.014, -0.05,  -0.082, -0.033]
# sigma_z = [
#     [ 0.001,  0.001, -0.001,  0.,     0.,     0.002],
#     [ 0.001,  0.,    -0.001,  0.,     0.,     0.   ],
#     [-0.001, -0.001,  0.002, -0.001, -0.001,  0.   ],
#     [ 0.   ,  0.,    -0.001,  0.001,  0.,    -0.   ],
#     [ 0.   ,  0.,    -0.001,  0.,     0.,    -0.   ],
#     [ 0.002,  0.,     0.,    -0.,    -0.,     0.005],
#  ]

# import matplotlib.pyplot as plt

# data = []
# for i in range(100000):
#     data.append(np.random.multivariate_normal(mu_z, sigma_z)[0])

# # Specify custom bins
# bins = np.arange(-15, 1) * 0.01

# # Plot histogram with custom bins
# plt.hist(data, bins=bins, edgecolor='black')

# # Add labels and title
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram with Custom Bins')

# # Display the plot
# plt.show()
# ================================================================


# # ==================== Make video from multiple videos ====================
# from moviepy.editor import VideoFileClip, concatenate_videoclips
# import glob

# path = "/home/arpit/test_projects/OmniGibson/outputs/2024-08-27/15-46-38"

# # Use glob to get a list of all .mp4 files in the directory
# video_files = sorted(glob.glob(os.path.join(path, "*.mp4")))

# # Alternatively, if you want to be explicit about sorting by filename:
# video_files = sorted(glob.glob(os.path.join(path, "*.mp4")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
# print("video_files: ", video_files)

# # Load the video clips
# clips = [VideoFileClip(video) for video in video_files]

# # Concatenate the video clips
# final_clip = concatenate_videoclips(clips)

# # Write the output video to a file
# final_clip.write_videofile("output_video.mp4", codec="libx264")
# =============================================================================


# # ============== Delete certain episodes from a hdf5 file ===============
# hdf5_file_path = '/home/arpit/test_projects/OmniGibson/pick_data/dataset.hdf5'
# file = h5py.File(hdf5_file_path, 'a')
# all_keys = list(file['data'].keys())

# for i, ep in enumerate(file['data'].keys()):
#     print("i, ep: ", i, ep)
#     if ep == 'data':        
#         print("deleting")
#         del file[f'data/{ep}']
# ===========================================================================
