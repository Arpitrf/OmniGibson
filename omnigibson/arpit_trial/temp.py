from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
import pickle

# # read episode errors
# with open("/home/arpit/test_projects/OmniGibson/outputs_data_gen/2024-11-13/22-18-29/episode_errors.pkl", "rb") as f:
#     episode_errors = pickle.load(f)
# # print the metadata of the first 10 episodes
# phases = []
# reasons = []
# for i in range(50):
#     print(f"Episode {i}: {episode_errors[i]['reason']}")
#     # print(f"Episode {i}: {episode_errors[i]['phase']}")
#     phases.append(episode_errors[i]['phase'])
#     reasons.append(episode_errors[i]['reason'])



# import matplotlib.pyplot as plt
# import numpy as np
# from collections import Counter
# # Create a dictionary to store phase-reason pairs
# phase_reason_pairs = {}
# for i, phase in enumerate(phases):
#     reason = reasons[i]
#     if phase not in phase_reason_pairs:
#         phase_reason_pairs[phase] = Counter()
#     phase_reason_pairs[phase][reason] += 1

# # Set up the plot
# fig, ax = plt.subplots(figsize=(12, 6))

# # Get unique reasons for the legend
# unique_reasons = set(reasons)
# colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique_reasons)))
# reason_color_map = dict(zip(unique_reasons, colors))

# # Plot stacked bars for each phase
# x = np.arange(len(phase_reason_pairs))
# bottom = np.zeros(len(phase_reason_pairs))

# for reason in unique_reasons:
#     heights = [phase_reason_pairs[phase][reason] for phase in phase_reason_pairs.keys()]
#     ax.bar(x, heights, bottom=bottom, label=reason, color=reason_color_map[reason])
    
#     # Add value labels for non-zero counts
#     for i, height in enumerate(heights):
#         if height > 0:
#             ax.text(x[i], bottom[i] + height/2, str(int(height)), 
#                    ha='center', va='center')
#     bottom += heights

# # Customize the plot
# ax.set_ylabel('Count')
# ax.set_title('Distribution of Errors by Phase and Reason')
# ax.set_xticks(x)
# ax.set_xticklabels(list(phase_reason_pairs.keys()), rotation=45, ha='right')
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.tight_layout()
# plt.show()


import h5py
import numpy as np
with h5py.File("/home/arpit/test_projects/OmniGibson/temp/dataset.hdf5", "r") as f:
    print(len(f["data"].keys()))
    # breakpoint()
    for i in range(len(f["data"])):
        print("actions: ", np.array(f["data"]["episode_{:05d}".format(i)]["actions"]["actions"]).shape)
        print("rgb: ", np.array(f["data"]["episode_{:05d}".format(i)]["observations"]["rgb"]).shape)
        print("contacts: ", np.array(f["data"]["episode_{:05d}".format(i)]["extras"]["contacts"]).shape)
        print("singularities: ", np.array(f["data"]["episode_{:05d}".format(i)]["extras"]["singularities"]).shape)
        print("extrinsic_matrix: ", np.array(f["data"]["episode_{:05d}".format(i)]["proprioceptions"]["extrinsic_matrix"]).shape)
        print(("--------------"))

#     print(np.array(f["data"]["episode_00250"]["actions"]["actions"]))
#     print(np.array(f["data"]["episode_00250"]["proprioceptions"]["right_eef_pos"]).shape)
#     print(np.array(f["data"]["episode_00250"]["extras"]["contacts"]))

    # # count the number of contacts in the entire dataset
    # total_data_points = 0
    # total_contacts = 0
    # for i in range(len(f["data"])):
    #     # if f["data"]["episode_{:05d}".format(i)]["extras"]["contacts"][0]:
    #     #     print(f"Episode {i} has contacts in the first timestep")
    #     if len(f["data"]["episode_{:05d}".format(i)]['extras']['contacts']) == 1:
    #         print(f"Ignoring episode {i}")
    #         continue
    #     for j in range(1, len(f["data"]["episode_{:05d}".format(i)]["extras"]["contacts"])):
    #         contact = f["data"]["episode_{:05d}".format(i)]["extras"]["contacts"][j]
    #         total_data_points += 1
    #         if contact:
    #             total_contacts += 1
    # print(f"Total data points: {total_data_points}, Total contacts: {total_contacts}")



# def filter_hdf5_episodes(input_path, output_path, episode_keys):
#     """
#     Read a hdf5 file and create a new hdf5 file containing only specified episodes.
    
#     Args:
#         input_path (str): Path to input hdf5 file
#         output_path (str): Path where filtered hdf5 file will be saved
#         episode_keys (list): List of episode keys to keep in filtered file
#     """
#     # Open input file in read mode
#     with h5py.File(input_path, 'r') as src:
#         # Create output file
#         with h5py.File(output_path, 'w') as dst:
#             # Copy over all attributes from source file
#             for key, val in src.attrs.items():
#                 dst.attrs[key] = val
            
#             # Copy data group
#             if 'data' in src:
#                 data_group = dst.create_group('data')
#                 # Only copy specified episodes
#                 for ep in episode_keys:
#                     if ep in src['data']:
#                         src.copy(f'/data/{ep}', data_group)
            
#             # Copy mask group if it exists
#             if 'mask' in src:
#                 mask_group = dst.create_group('mask')
#                 # Copy all mask datasets
#                 for key in src['mask'].keys():
#                     # Get original mask data
#                     orig_mask = src['mask'][key][:]
#                     # Filter to only include specified episodes
#                     filtered_mask = [x for x in orig_mask if x.decode('utf-8') in episode_keys]
#                     # Create new dataset with filtered mask
#                     mask_group.create_dataset(key, data=filtered_mask)

#     return

# filter_hdf5_episodes("/home/arpit/test_projects/OmniGibson/place_in_shelf_data/dataset.hdf5", "/home/arpit/test_projects/OmniGibson/place_in_shelf_data/filtered_dataset.hdf5", ["episode_00250"])