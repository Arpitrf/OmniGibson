import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import wandb

def is_number(s):
    try:
        # Try converting the string to an integer or float
        float(s)
        return True
    except ValueError:
        # If conversion fails, return False
        return False

def display_segmentation_image(segmentation_image):
    # Get unique labels in the segmentation image
    unique_labels = np.unique(segmentation_image)
    print("unique_labels: ", unique_labels)
    mapping = {
        825831922: 1,
        2814990211: 2,
        2848186650: 3, 
        3157253163: 4,
        4013582681: 5
    }
    for i in  range(segmentation_image.shape[0]):
        for j in range(segmentation_image.shape[1]):
            segmentation_image[i][j] = mapping[segmentation_image[i][j]]

    seg_display = segmentation_image
    return seg_display

    # # Create a colormap with distinct colors for each label
    # num_labels = len(unique_labels)
    # colormap = plt.cm.get_cmap('hsv', num_labels)
    # colors = [colormap(i) for i in range(num_labels)]
    # segmented_cmap = ListedColormap(colors)

    # # Display the segmentation image with the assigned colors
    # plt.imshow(segmentation_image, cmap=segmented_cmap)
    # plt.colorbar()
    # plt.show()

wandb_log = True
save_video = False
contact_frames_table = None
if wandb_log:
    wandb.login()
    run = wandb.init(
        project="VMPC",
        notes="trial experiment"
    )
    table_1 = wandb.Table(columns=["f_name", "Noisy Video", "Teleop Traj Last Img", "Noisy Traj Last Img", "Cost", "Success"])
    # contact_points_table = wandb.Table(columns=["Video Name", "Left Hand Contact Points", "Right Hand Contact Points"])

goal_path = f'output_assisted/goal_traj/0/obs'
f_names = []
for f_name in os.listdir(goal_path):
    f_names.append(f_name)
f_names.sort(key=lambda x: int(x.split('.')[0]))
goal_img_f_name = f_names[-1].split('.')[0]
with open(f'{goal_path}/{f_names[-1]}', 'rb') as f:
    obs_goal = pickle.load(f)

rows = []
folders = ['noisy_1', 'noisy_2', 'succ', 'goal_traj']
# folders = ['noisy_1', 'succ']
for folder in folders:
    directory = f'output_assisted/{folder}'
    # Load other trajs
    for name in os.listdir(directory):
        print("name: ", name)
        # if is_number(name):
        if name != 'fetch':
            path = f'{directory}/{name}/obs'
            f_names = []
            for f_name in os.listdir(path):
                f_names.append(f_name)
            f_names.sort(key=lambda x: int(x.split('.')[0]))

            with open(f'{path}/{f_names[-1]}', 'rb') as f:
                obs = pickle.load(f)

            # load traj info
            with open(f'{directory}/{name}/traj_info.pickle', 'rb') as f:
                traj_info = pickle.load(f)
                obj_in_hand = traj_info['obj_in_hand']
                print("traj_info: ", traj_info['obj_in_hand'])        

            # compute the cost
            seg_ideal = obs_goal['robot0']['robot0:eyes:Camera:0']['seg_semantic']
            rgb_ideal = obs_goal['robot0']['robot0:eyes:Camera:0']['rgb']
            seg = obs['robot0']['robot0:eyes:Camera:0']['seg_semantic']
            rgb = obs['robot0']['robot0:eyes:Camera:0']['rgb']
            total_pixels = seg.shape[0] * seg.shape[1]
            pixel_cost = np.sum(seg_ideal != seg) / total_pixels
            print("pixel_cost: ", pixel_cost)      

            file_name = f_names[-1].split('.')[0]
            # rows.append([wandb.Video(f'{directory}/ideal/video_seg_display.mp4', fps=30, format="mp4"), 
            rows.append([f'{folder}/{name}',
                            wandb.Video(f'{directory}/{name}/video_seg_display.mp4', fps=30, format="mp4"),
                            wandb.Image(f'output_assisted/goal_traj/0/seg_display/{goal_img_f_name}.jpg'),
                            wandb.Image(f'{directory}/{name}/seg_display/{file_name}.jpg'),
                            pixel_cost,
                            obj_in_hand])   

        # display_segmentation_image(seg_ideal)
        # fig, ax = plt.subplots(2,2)
        # ax[0][0].imshow(seg_ideal)
        # ax[0][1].imshow(seg)
        # ax[1][0].imshow(rgb_ideal)
        # ax[1][1].imshow(rgb)
        # plt.show()

# print("costs: ", len(rows), type(rows))
if wandb_log:
    table_1 = wandb.Table(
        columns=table_1.columns, data=rows
    )
    run.log({"Cost": table_1}) 
         
        
