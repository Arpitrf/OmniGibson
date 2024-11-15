import os
import imageio
import glob
def compress_video(input_path, output_path, target_size_mb=100):
    """
    Compresses a video file to target size in MB while maintaining aspect ratio.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path where compressed video will be saved 
        target_size_mb (int): Desired size in MB for output video
    """
    # Get input video size in bytes
    input_size = os.path.getsize(input_path) / (1024 * 1024)  # Convert to MB
    
    if input_size <= target_size_mb:
        print(f"Input video ({input_size:.1f}MB) is already smaller than target size ({target_size_mb}MB)")
        return
        
    # Calculate target bitrate (rule of thumb: bitrate = file size / duration)
    reader = imageio.get_reader(input_path)
    duration = reader.count_frames() / reader.get_meta_data()['fps']
    target_bitrate = int((target_size_mb * 8192) / duration)  # Convert to kbps
    
    # Compress using lower quality and bitrate
    writer = imageio.get_writer(
        output_path,
        fps=reader.get_meta_data()['fps'],
        quality=5,  # Lower quality (1-10)
        bitrate=target_bitrate,
        codec='h264',
        macro_block_size=None,
        ffmpeg_params=['-vf', 'scale=iw/2:ih/2']  # Reduce resolution by half
    )
    
    for frame in reader:
        writer.append_data(frame)
        
    reader.close()
    writer.close()
    
    # Print compression results
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Compressed video size: {output_size:.1f}MB (reduced from {input_size:.1f}MB)")

def combine_episode_videos(folder_path, output_path):
    """
    Combines multiple episode videos into a single video file.
    
    Args:
        folder_path (str): Path to folder containing episode videos
        output_path (str): Path where combined video will be saved
    """
    # Get list of episode video files sorted numerically
    video_files = sorted(glob.glob(os.path.join(folder_path, "episode_*.mp4")))
    
    if not video_files:
        print(f"No episode videos found in {folder_path}")
        return
        
    # Read the first video to get metadata
    first_video = imageio.get_reader(video_files[0])
    fps = first_video.get_meta_data()['fps']
    
    # Set up video writer with same parameters as input videos
    writer = imageio.get_writer(output_path, fps=fps)
    
    # Combine all videos
    for video_path in video_files:
        print(f"Processing {os.path.basename(video_path)}...")
        reader = imageio.get_reader(video_path)
        for frame in reader:
            writer.append_data(frame)
        reader.close()
            
    writer.close()
    print(f"Combined video saved to {output_path}")

from moviepy.editor import VideoFileClip, vfx

# Load the video file
input_path = "/home/arpit/test_projects/OmniGibson/outputs_data_gen/2024-11-13/20-07-11/combined_video.mp4"  # Replace with your video file path
output_path = "/home/arpit/test_projects/OmniGibson/outputs_data_gen/2024-11-13/20-07-11/combined_video_fast.mp4"  # Output file path
speed_factor = 8.0  # Factor by which to speed up the video

# Load the video clip
clip = VideoFileClip(input_path)

# Speed up the clip
sped_up_clip = clip.fx(vfx.speedx, speed_factor)

# Write the sped-up video to the output file
sped_up_clip.write_videofile(output_path, codec="libx264", fps=clip.fps)

print(f"Video saved to {output_path}")

# combine_episode_videos("/home/arpit/test_projects/OmniGibson/outputs_data_gen/2024-11-13/22-18-29", 
#                        "/home/arpit/test_projects/OmniGibson/outputs_data_gen/2024-11-13/22-18-29/combined_video.mp4")