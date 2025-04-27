import cv2
import os

def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Hata: {video_path} yolu açılmadı.")
        exit()
    return cap

def get_video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, total_frames

def create_log_directory():
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    log_dir = f"logs/{today}"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir
