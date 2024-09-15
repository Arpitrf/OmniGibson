import h5py
from filelock import FileLock
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import random
import hdf5plugin
import pdb
import torch as th

class Memory:
    log = False
    base_keys = [
        'observations',
        'observations_info',
        'actions',
        'proprioceptions',
        'extras'
    ]
    observation_keys = [
        'rgb',
        'depth',
        'normal',
        'seg_semantic',
        'seg_instance',
        'seg_instance_id',
        'gripper_obj_seg'
    ]
    proprioception_keys = ['joint_qpos', 'joint_qpos_sin', 'joint_qpos_cos', 'joint_qvel', 'joint_qeffort', 'robot_pos', 'robot_ori_cos', 'robot_ori_sin', 'robot_2d_ori', 'robot_2d_ori_cos', 'robot_2d_ori_sin', 'robot_lin_vel', 'robot_ang_vel', 'camera_qpos', 'camera_qpos_sin', 'camera_qpos_cos', 'camera_qvel', 'base_qpos', 'base_qpos_sin', 'base_qpos_cos', 'base_qvel', 'arm_left_qpos', 'arm_left_qpos_sin', 'arm_left_qpos_cos', 'arm_left_qvel', 'eef_left_pos_global', 'eef_left_quat_global', 'eef_left_pos', 'eef_left_quat', 'grasp_left', 'gripper_left_qpos', 'gripper_left_qvel', 'arm_right_qpos', 'arm_right_qpos_sin', 'arm_right_qpos_cos', 'arm_right_qvel', 'eef_right_pos_global', 'eef_right_quat_global', 'eef_right_pos', 'eef_right_quat', 'grasp_right', 'gripper_right_qpos', 'gripper_right_qvel', 'trunk_qpos', 'trunk_qvel', 'left_eef_pos', 'left_eef_orn', 'right_eef_pos', 'right_eef_orn', 'base_pos', 'base_orn']
    action_keys = [        
        'complete_actions',
        'preprocessed_actions',
        'processed_actions',
        'actions'
    ]
    extra_keys = [
        'grasps',
        'contacts',
    ]
    observation_info_keys = [
        'seg_semantic',
        'seg_instance',
        'seg_instance_id',
    ]
    # base_keys = [
    #     'rgbs',
    #     'depths',
    #     'segs_semantic',
    #     'normals',
    #     'segs_instance',
    #     'complete_actions',
    #     'preprocessed_actions',
    #     'processed_actions',
    #     'grasps',
    #     'contacts',
    #     'proprioceptions'
    # ]


    def __init__(self):
        self.data = {}
        for key in Memory.base_keys:
            print("keyy: ", key)
            self.data[key] = {}
        for key in Memory.observation_keys:
            self.data['observations'][key] = []
        for key in Memory.observation_info_keys:
            self.data['observations_info'][key] = []
        for key in Memory.action_keys:
            self.data['actions'][key] = []
        for key in Memory.proprioception_keys:
            self.data['proprioceptions'][key] = []
        for key in Memory.extra_keys:
            self.data['extras'][key] = []

    # @staticmethod
    # def concat(memories):
    #     output = Memory()
    #     for memory in memories:
    #         for key in memory.data:
    #             if key not in output.data:
    #                 output.data[key] = []
    #             output.data[key].extend(memory.data[key])
    #     return output

    # def clear(self):
    #     for key in self.data:
    #         del self.data[key][:]

    # def print_length(self):
    #     output = "[Memory] "
    #     for key in self.data:
    #         output += f" {key}: {len(self.data[key])} |"
    #     print(output)

    # def assert_length(self):
    #     key_lens = [len(self.data[key]) for key in self.data]

    #     same_length = key_lens.count(key_lens[0]) == len(key_lens)
    #     if not same_length:
    #         self.print_length()

    def __len__(self):
        return len(self.data['observations']['rgb'])

    # def add_rewards_and_termination(self, reward, termination):
    #     assert len(self.data['rewards']) \
    #         == len(self.data['is_terminal'])\
    #         == len(self.data['actions']) - 1\
    #         == len(self.data['observations']) - 1
    #     self.data['rewards'].append(float(reward))
    #     self.data['is_terminal'].append(float(termination))

    def add_observation(self, key, val):
        # assert len(self.data['rewards']) \
        #     == len(self.data['is_terminal'])\
        #     == len(self.data['actions'])\
        #     == len(self.data['observations'])
        self.data['observations'][key].append(deepcopy(val))

    def add_observation_info(self, key, val):
        # print(key, val)
        # input()
        arr = []
        for k, v in val.items():
            arr.append([str(k), v])
        # print("arr: ", arr)
        self.data['observations_info'][key].append(deepcopy(arr))

    def add_action(self, key, val):
        self.data['actions'][key].append(deepcopy(val))

    def add_proprioception(self, key, val):
        self.data['proprioceptions'][key].append(deepcopy(val))

    def add_extra(self, key, val):
        self.data['extras'][key].append(deepcopy(val))

    def add_value(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(deepcopy(value))

    def has_variable_length_array(self, array):
        def check_shapes(arr):
            # Convert the array to a NumPy array with dtype=object to handle varying sizes
            np_array = np.array(arr, dtype=object)

            # Get the shapes of all elements at this level
            shapes = [np.shape(sub_array) for sub_array in np_array]
            
            # If there is more than one unique shape, the array is of variable length
            if len(set(shapes)) > 1:
                return True
            
            # Recursively check the next level if it's deeper than a single level (not scalar)
            for sub_array in np_array:
                if isinstance(sub_array, np.ndarray) or isinstance(sub_array, list):
                    if check_shapes(sub_array):
                        return True
            
            return False
        
        return check_shapes(array)
        
    def add_variable_length_dataset(self, group, key, value):
        # Flatten the list
        flattened_data = [item for sublist in value for inner in sublist for item in inner]

        # Track the shapes to reconstruct the structure
        shapes = [len(sublist) for sublist in value]

        # Create a special dtype for variable-length strings
        vlen_dtype = h5py.special_dtype(vlen=str)
       
        # store the actual data (which is now flattened hence, no uneven shapes) 
        group.create_dataset(f'{key}_strings', data=np.array(flattened_data, dtype=vlen_dtype), compression='gzip', compression_opts=9)
        
        # Store the shapes (which will be used to reconstruct the uneven array)
        group.create_dataset(f'{key}_shapes', data=np.array(shapes, dtype=np.int32), compression='gzip', compression_opts=9)
    
    def dump_recursive(self, group, key, value):
        # print("key, type(value): ", key, type(value))
        # pdb.set_trace()
        # if key == 'seg_semantic_id':
        #     print("value: ", value, type(value[0][0]))
        try:
            if type(value) == float\
                    or type(value) == np.float64\
                    or type(value) == str\
                    or type(value) == int\
                    or type(value) == np.int64\
                    or type(value) == bool:
                # print("11", key)
                group.attrs[key] = value
            elif type(value) == list or type(value) == np.ndarray or type(value) == tuple:   
                # if len(value) != 0 and type(value[0][0][0]) == str:
                #     print("1111111")
                #     value = value
                # elif len(value) != 0:
                #     print("222222222222")
                #     value = th.stack(value, dim=0).cpu().numpy()
                # value = value.astype(np.float64)
                # print("22", key)
                # HDF5 Can't save variable length arrays/lists. i.e. say shape is (9,) and the first element has len=7 and second element has len=6
                variable_length = self.has_variable_length_array(np.array(value))
                if variable_length:
                    self.add_variable_length_dataset(group, key, value)
                else:
                    group.create_dataset(
                        name=key,
                        data=value,
                        # **hdf5plugin.Bitshuffle(nelems=0, lz4=True)
                        compression='gzip',
                        compression_opts=9)
            else:
                # print("33", key)
                subgroup = group.create_group(key)
                for key_interior, value_interior in value.items():
                    # subgroup = group.create_group(key_interior)
                    # subgroup = group
                    # if type(value_interior) == dict:
                    #     subgroup = group.create_group(key_interior)
                    self.dump_recursive(subgroup, key_interior, value_interior)
        except Exception as e:
             raise Exception(f'[Memory] Dump key {key} error with value type {type(value)}:', e)
            # print(value)

    def dump(self, hdf5_path, log=False):
        # for k, val in self.data.items():
        #     print("--k, val: ", k, len(val))
        with FileLock(hdf5_path + ".lock"):
            with h5py.File(hdf5_path, 'a') as file:
                key_idx = 0
                if 'data' in file.keys():
                    key_idx = len(file['data'].keys())
                else:
                    file.create_group('data')
                # print("key_idx: ", key_idx)
                group_key = f'episode_{key_idx:05d}'
                group = file['data'].create_group(group_key)
                # print("file.keys(): ", file.keys())

                for key, value in self.data.items():
                    # print("key: ", key)                 
                    self.dump_recursive(group, key, value)

                # for step in range(len(self)):
                #     step_key = f'step_{step:05d}'
                #     group_step = group.create_group(step_key)

                #     for key, value in self.data.items():
                #         # print("k: ", key)
                #         step_value = value[step]                    
                #         self.dump_recursive(group_step, key, step_value)
                # print("group: ", group)
                
