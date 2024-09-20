import h5py
import numpy as np
np.set_printoptions(suppress=True, precision=3)

f = h5py.File('moma_pick_and_place/dataset.hdf5', "r")
# ======================== Basic hdf5 testing ======================
# f = h5py.File('prior/dataset.hdf5', "r")
print("len: ", len(f['data']))
print("--", f['data/episode_00000/observations'].keys())
print("--", np.array(f['data/episode_00000/observations/rgb']).shape)
print(np.array(f['data/episode_00000/actions/actions']))
# print("--", np.array(f[f'data/episode_00000/observations_info']['seg_semantic']))
# print("joint efforts: ", np.array(f[f'data/episode_00000/proprioceptions/joint_qeffort']).shape)
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