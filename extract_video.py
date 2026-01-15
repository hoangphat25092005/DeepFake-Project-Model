import cv2
import os
from pathlib import Path

def extract_frames(video_folder, output_folder, sample_rate=30):
    os.makedirs(output_folder, exist_ok=True)

    video_files = list(Path(video_folder).glob('*.mp4')) + list(Path(video_folder).glob('*.avi')) + list(Path(video_folder).glob('*.mov'))

    print(f"Found {len(video_files)} video files in {video_folder}")

    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        video_name = video_path.stem
        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                frame_filename = f"{video_name}_frame_{extracted_count:06d}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1

            frame_count += 1

        cap.release()
        print(f"{video_name}: Extracted {extracted_count} frames from {frame_count} total frames.")
    
    print(f"\nTotal frames extracted: {len(list(Path(output_folder).glob('*.jpg')))}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from videos in a folder.")
    parser.add_argument('--video_folder', type=str, required=True, help='Path to the folder containing video files.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the folder to save extracted frames.')
    parser.add_argument('--sample_rate', type=int, default=30, help='Extract one frame every N frames.')

    args = parser.parse_args()

    extract_frames(args.video_folder, args.output_folder, args.sample_rate)

