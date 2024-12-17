import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, edgecolor='green'):
    x0, y0 = box[:, 0], box[:, 1]
    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))  

def obtain_mask():
    # sam
    CHECKPOINT_PATH='/home/arpit/test_projects/segment-anything/sam_vit_h_4b8939.pth'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"


    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Give the path of your image
    f_name = '0001'
    IMAGE_PATH= f"/home/arpit/test_projects/OmniGibson/real_world_data/{f_name}_rgb.png"
    image= cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(image_rgb)

    # input_point = np.array([
    #         [530, 350],
    #         [106, 428],
    #         [154, 194],
    #         [24, 253],
    #         [214, 318],
    #         [298, 395],
    #         [616, 303]
    #     ])

    # # episode_0000
    # prompts = {
    #     # "robot": np.array([530, 350]),
    #     "robot": np.array([[24, 253, 628, 460]]),
    #     "box": np.array([[62, 430]]),
    #     "shelf": np.array([[144, 194], [163, 348]]),
    # } 
    # episode_0001
    prompts = {
        # "robot": np.array([530, 350]),
        "robot": np.array([[24, 119, 628, 410]]),
        "box": np.array([[46, 346], [45, 437]]),
        "shelf": np.array([[144, 194], [163, 348]]),
    }
    final_masks = []
    VISUALIZE = True
    for prompt_name, prompt_coords in prompts.items():
        if VISUALIZE:
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            if len(prompt_coords[0]) == 2:
                input_label = np.ones(len(prompt_coords))
                show_points(prompt_coords, input_label, plt.gca())
            else:
                show_box(prompt_coords, plt.gca())    
            plt.axis('on')
            plt.show()

        if len(prompt_coords[0]) == 2:
            input_label = np.ones(len(prompt_coords))
            masks, scores, logits = mask_predictor.predict(
                point_coords=prompt_coords,
                point_labels=input_label,
                multimask_output=True,
            ) 
        else:
            masks, scores, logits = mask_predictor.predict(
                box=prompt_coords,
                multimask_output=True,
            ) 

        # masks shape: (3, 600, 800): (number_of_masks) x H x W
        # Sort masks and scores by score value in descending order
        sorted_indices = np.argsort(scores)[::-1]  # Get indices that would sort scores in descending order
        masks = masks[sorted_indices]
        scores = scores[sorted_indices]
        logits = logits[sorted_indices]

        final_masks.append(masks[0])

        if VISUALIZE:
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10,10))
                plt.imshow(image)
                if len(prompt_coords[0]) == 2:
                    input_label = np.ones(len(prompt_coords))
                    show_points(prompt_coords, input_label, plt.gca())
                else:
                    show_box(prompt_coords, plt.gca())   
                show_mask(mask, plt.gca())
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                plt.axis('off')
                plt.show()

    # Combine all masks using logical OR operation
    combined_mask = np.zeros_like(final_masks[0], dtype=bool)
    for mask in final_masks:
        combined_mask = np.logical_or(combined_mask, mask)

    # Visualize the combined mask
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(combined_mask, plt.gca())
    plt.title("Combined Mask", fontsize=18)
    plt.axis('off')
    plt.show()

    return combined_mask

if __name__ == "__main__":
    obtain_mask()