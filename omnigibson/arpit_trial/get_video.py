import cv2
import os

def create_video_from_images(image_folder, output_video, fps=30):
    # Get list of all files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    images.sort(key=lambda x: int(os.path.splitext(x)[0]))  # For OmniGibson videos
    # images.sort(key=lambda x: int(x.split("_")[1])) # For HaMer videos
    # images.pop(0)

    if not images:
        print("No .jpg images found in the folder.")
        return

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, 0, fps, (width, height))

    list = os.listdir(image_folder)
    # print(images)

    # Loop through all images and write them to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video {output_video} created successfully.")

# Example usage:
# image_folder = 'prior_test1_images'
# output_video = f'{image_folder}.avi'
# create_video_from_images(image_folder, output_video, fps=15)
