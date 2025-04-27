from config import load_config, is_cuda_available
from logger import setup_logger
from model_loader import load_model
from video_handler import open_video, get_video_info, create_log_directory
from processing import process_video
from utils import get_video_start_time
import time
import numpy as np

def main():
    setup_logger()
    config = load_config()

    use_cuda = is_cuda_available()
    model = load_model(config.get("model_path", "models/yolov8x.pt"), use_cuda)
    print(f"Model şu cihazda çalışıyor: {'GPU (CUDA)' if use_cuda else 'CPU'}")

    cap = open_video(config.get("video_path", "videos/demo4.mp4"))
    fps, total_frames = get_video_info(cap)
    print(f"Video süresi: {int(total_frames / fps)} saniye ≈ {int((total_frames / fps) // 60)} dakika")

    working_zone = config.get("working_zone", [[700, 50], [1200, 50], [1200, 710], [700, 710]])
    working_zone = np.array(working_zone)

    log_dir = create_log_directory()
    video_start_time = get_video_start_time()

    start_time = time.time()

    processed_frames = process_video(
        cap,
        model,
        working_zone,
        config.get("conf_threshold", 0.5),
        config.get("iou_threshold", 0.5),
        config.get("batch_size", 1),
        video_start_time,
        log_dir
    )

    end_time = time.time()
    elapsed = end_time - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    print("\nAnaliz tamamlandı.")
    print(f"Toplam analiz edilen kare sayısı: {processed_frames}")
    print(f"Log klasörü: {log_dir}")
    print(f"Analiz süresi: {minutes} dakika {seconds} saniye")

if __name__ == "__main__":
    main()
