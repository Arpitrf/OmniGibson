import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

def combine_videos(folder_path):
    # Get the list of all mp4 files in the folder
    video_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mp4')])

    # Load all videos as VideoFileClip objects
    clips = [VideoFileClip(os.path.join(folder_path, file)) for file in video_files]

    # Concatenate all the video clips
    final_clip = concatenate_videoclips(clips)

    # Write the result to a file
    output_file = "combined_video.mp4"
    final_clip.write_videofile(output_file)

    print(f"Combined video saved as {output_file}")

path = "outputs/run_2024-09-22_16-25-42"
combine_videos(path)
