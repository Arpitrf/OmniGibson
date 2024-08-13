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

wandb_log = True
save_video = False
contact_frames_table = None
if wandb_log:
    wandb.login()
    run = wandb.init(
        project="VMPC",
        notes="trial experiment"
    )
    table_1 = wandb.Table(columns=["f_name", "LIV Result", "Goal Last Img", "Curr Last Img", "Cost Goal Image", "Cost Goal Text", "Success"])
    # contact_points_table = wandb.Table(columns=["Video Name", "Left Hand Contact Points", "Right Hand Contact Points"])


# Load the ideal traj obs
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
        path = f'{directory}/{name}/obs'
        f_names = []
        for f_name in os.listdir(path):
            f_names.append(f_name)
        f_names.sort(key=lambda x: int(x.split('.')[0]))

        with open(f'{directory}/{name}/liv_distances.pickle', 'rb') as f:
            liv_distances = pickle.load(f)

        # load traj info
        with open(f'{directory}/{name}/traj_info.pickle', 'rb') as f:
            traj_info = pickle.load(f)
            obj_in_hand = traj_info['obj_in_hand']
            print("traj_info: ", traj_info['obj_in_hand'])        

        # compute the cost
        last_img_cost = liv_distances['text'][-1]
        last_10_img_cost = np.mean(liv_distances['text'][-10:])
        img_goal_cost = liv_distances['image'][-1]
        print("LIV cost: ", last_img_cost, last_10_img_cost)      

        file_name = f_names[-1].split('.')[0]

        rows.append([f'{folder}/{name}',
                    wandb.Video(f'{directory}/{name}/liv_video.mp4', fps=30, format="mp4"),
                    wandb.Image(f'output_assisted/goal_traj/0/rgb/{goal_img_f_name}.jpg'),
                    wandb.Image(f'{directory}/{name}/rgb/{file_name}.jpg'),
                    img_goal_cost,
                    last_img_cost, 
                    # last_10_img_cost,
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
         
        
