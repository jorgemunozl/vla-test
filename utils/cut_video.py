# Script that takes a video and cuts it into N seconds.
import cv2
import numpy as np
import subprocess

subprocess.run(["pwd"], capture_output=True, text=True)

def cut_video(video_path, output_path, seconds):
    """Cut a video into N seconds chunks using ffmpeg."""
    subprocess.run(["ffmpeg", "-i", video_path, "-c", "copy", "-map", "0", "-segment_time", str(seconds), "-f", "segment", f"{output_path}/segment_%03d.mp4"])
    print(f"Video cut into {seconds} seconds chunks and saved to {output_path}")

if __name__ == "__main__":
    video_path = "./utils/video.mp4"
    output_path = "./utils/cut_video_output.mp4"
    seconds = 14
    cut_video(video_path, output_path, seconds)
    print(f"Video cut into {seconds} seconds chunks and saved to {output_path}")