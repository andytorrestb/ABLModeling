import cv2, os

def save_video(output_dir, video_filename="output.mp4", fps=10):
    os.makedirs(output_dir, exist_ok=True)
    # Get all image files in the directory (e.g., png, jpg)
    images = [img for img in sorted(os.listdir(output_dir)) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found in the directory.")
        return

    # Read the first image to get frame size
    first_frame = cv2.imread(os.path.join(output_dir, images[0]))
    height, width, layers = first_frame.shape

    # Define the video writer
    video_path = os.path.join(output_dir, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write each image as a frame
    for img_name in images:
        frame = cv2.imread(os.path.join(output_dir, img_name))
        video.write(frame)

    video.release()
    # print(f"Video saved as {video_path}")

if __name__ == "__main__":
    fps = 30
    case_directory = "output_Re_100000_L_14/"
    vel_output_dir = os.path.join(case_directory, "vel_magnitude_output/")
    vor_output_dir = os.path.join(case_directory, "vorticity_output/")
    save_video(vel_output_dir, fps=fps)
    save_video(vor_output_dir, fps=fps)