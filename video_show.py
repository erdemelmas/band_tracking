import cv2
import numpy as np
from ultralytics import YOLO
import statistics
import csv
import os
import json
from datetime import datetime, timedelta
from ultralytics.utils import LOGGER
import torch
import time
from tqdm import tqdm

# Logger ayarı
LOGGER.setLevel("ERROR")

# Konfigürasyon dosyasını oku
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("❌ Hata: config.json dosyası bulunamadı!")
    exit()
except json.JSONDecodeError:
    print("❌ Hata: config.json dosyası okunamadı (format hatalı).")
    exit()

use_cuda = torch.cuda.is_available()
model_path = config.get("model_path", "models/yolov8x.pt")
video_path = config.get("video_path", "videos/demo4.mp4")
conf_threshold = config.get("conf_threshold", 0.5)
iou_threshold = config.get("iou_threshold", 0.5)
batch_size = config.get("batch_size", 1)

model = YOLO(model_path).to('cuda') if use_cuda else YOLO(model_path)
print(f"🚀 Model şu cihazda çalışıyor: {'GPU (CUDA)' if use_cuda else 'CPU'}")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Hata: {video_path} yolu açılmadı.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration_sec = total_frames / fps
video_duration_min = int(video_duration_sec // 60)
print(f"🎞️ Video süresi: {int(video_duration_sec)} saniye ≈ {video_duration_min} dakika")

frame_count = 0
working_zone = np.array([[400, 200], [1200, 200], [1200, 710], [400, 710]])

# Zaman ve log klasörü
video_start_time = datetime.strptime("09:00", "%H:%M")
today = datetime.now().strftime("%Y-%m-%d")
log_dir = f"logs/{today}"
os.makedirs(log_dir, exist_ok=True)

minute_log_path = os.path.join(log_dir, "dakika_logu.csv")
chunk_log_path = os.path.join(log_dir, "5saniye_logu.csv")
stable_chunk_log_path = os.path.join(log_dir, "stabil_chunk_log.csv")

for path, headers in [(minute_log_path, ['saat_araligi', 'kisi_sayisi']),
                      (chunk_log_path, ['zaman_araligi', 'kisi_sayisi']),
                      (stable_chunk_log_path, ['zaman_araligi', 'kisi_sayisi'])]:
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(headers)

def log_minute_average(avg, minute_index):
    start = video_start_time + timedelta(minutes=minute_index)
    end = start + timedelta(minutes=1)
    zaman = f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
    with open(minute_log_path, 'a', newline='') as f:
        csv.writer(f).writerow([zaman, avg])

def log_chunk_average(avg, start_second):
    start = video_start_time + timedelta(seconds=start_second)
    end = start + timedelta(seconds=5)
    zaman = f"{start.strftime('%H:%M:%S')}-{end.strftime('%H:%M:%S')}"
    with open(chunk_log_path, 'a', newline='') as f:
        csv.writer(f).writerow([zaman, avg])

def log_stable_chunk(start_sec, end_sec, count):
    start = video_start_time + timedelta(seconds=start_sec)
    end = video_start_time + timedelta(seconds=end_sec)
    zaman = f"{start.strftime('%H:%M:%S')}-{end.strftime('%H:%M:%S')}"
    with open(stable_chunk_log_path, 'a', newline='') as f:
        csv.writer(f).writerow([zaman, count])

start_time = time.time()

second_level_counts = []
chunk_averages = []
current_minute_index = 0
processed_frames = 0
total_logs_written = 0
chunk_start_time_sec = 0

stable_group = []
stable_group_start = None
frames_batch = []

# Progress Bar başlat
pbar = tqdm(total=total_frames, desc="🔍 Video işleniyor", unit="frame")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Her saniye 1 Kare çekmek için
        # if frame_count % int(fps) != 0:
        #     frame_count += 1
        #     pbar.update(1)
        #     continue

        frames_batch.append(frame)

        if len(frames_batch) == batch_size:
            results = model(frames_batch, classes=[0], conf=conf_threshold, iou=iou_threshold, verbose=False)
            for idx, result in enumerate(results):
                inside_count = 0
                # Çizim için kopya al
                display_frame = frames_batch[idx].copy()

                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    is_inside = cv2.pointPolygonTest(working_zone, center, False)
                    if is_inside >= 0:
                        # İçerideyse YEŞİL kutu ve YEŞİL nokta
                        inside_count += 1
                        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.circle(display_frame, center, 3, (0, 255, 0), -1)
                    else:
                        # Dışarıdaysa KIRMIZI kutu ve KIRMIZI nokta
                        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.circle(display_frame, center, 3, (0, 0, 255), -1)



                # Çalışma bölgesi çiz
                cv2.polylines(display_frame, [working_zone], isClosed=True, color=(255, 0, 0), thickness=2)
                # Kişi sayısını yaz
                cv2.putText(display_frame, f"Kisi (Zone): {inside_count}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Göster
                cv2.imshow("Video Analiz", display_frame)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

                second_level_counts.append(inside_count)
                processed_frames += 1

                if len(second_level_counts) == 5:
                    avg_5s = round(statistics.mean(second_level_counts))
                    chunk_averages.append(avg_5s)
                    log_chunk_average(avg_5s, chunk_start_time_sec)

                    if not stable_group:
                        stable_group = [avg_5s]
                        stable_group_start = chunk_start_time_sec
                    elif avg_5s == stable_group[-1]:
                        stable_group.append(avg_5s)
                    else:
                        if len(stable_group) >= 3:
                            log_stable_chunk(stable_group_start, chunk_start_time_sec, stable_group[0])
                        stable_group = [avg_5s]
                        stable_group_start = chunk_start_time_sec

                    chunk_start_time_sec += 5
                    second_level_counts = []

                elapsed_seconds = int(frame_count / fps)
                this_minute_index = elapsed_seconds // 60

                if this_minute_index != current_minute_index:
                    if chunk_averages:
                        avg_minute = round(statistics.mean(chunk_averages))
                        log_minute_average(avg_minute, current_minute_index)
                        total_logs_written += 1
                    current_minute_index = this_minute_index
                    chunk_averages = []

                pbar.update(1)

            frames_batch = []
        frame_count += 1

    # Kalan batch'ı işle
    if frames_batch:
        results = model(frames_batch, classes=[0], conf=conf_threshold, iou=iou_threshold, verbose=False)
        for idx, result in enumerate(results):
            inside_count = 0
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                is_inside = cv2.pointPolygonTest(working_zone, center, False)
                if is_inside >= 0:
                    inside_count += 1
            second_level_counts.append(inside_count)
            processed_frames += 1
            pbar.update(1)

finally:
    cap.release()
    cv2.destroyAllWindows()
    pbar.close()

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print("\n✅ Analiz tamamlandı.")
print(f"🧾 Toplam analiz edilen kare sayısı: {processed_frames}")
print(f"🕑 Log yazılan dakika sayısı: {total_logs_written}")
print(f"📁 Log klasörü: {log_dir}")
print(f"⏱️ Analiz süresi: {minutes} dakika {seconds} saniye")
