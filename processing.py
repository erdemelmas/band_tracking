import statistics
import cv2
import json
from datetime import timedelta
from tqdm import tqdm
from video_handler import get_video_info
import os
from display_handler import show_frame, close_window, draw_detections

# === Kullanıcı Ayarları ===
SHOW_VIDEO = True          # Videoyu ekranda göstermek için True
DRAW_DETECTIONS = True     # Kutu ve çalışma alanı çizmek için True
NORMAL_SPEED = True        # Normal hızda işlemek için True (False = Her saniye 1 Frame)
# ==========================

def point_inside_polygon(box, working_zone):
    x1, y1, x2, y2 = box.xyxy[0]
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    return cv2.pointPolygonTest(working_zone, center, False) >= 0

def log_chunk(log_json, start_sec, end_sec, count, video_start_time, degisim_var=False):
    start = video_start_time + timedelta(seconds=start_sec)
    end = video_start_time + timedelta(seconds=end_sec)
    zaman = f"{start.strftime('%H:%M:%S')}-{end.strftime('%H:%M:%S')}"
    entry = {
        "time_range": zaman,
        "person_count": count
    }
    if degisim_var is not None:
        entry["change_detected"] = degisim_var
    log_json.append(entry)

def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def boxes_to_centers(boxes, working_zone):
    centers = []
    for box in boxes:
        if point_inside_polygon(box, working_zone):
            x1, y1, x2, y2 = box.xyxy[0]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            centers.append(center)
    return centers

def match_center(center, center_list, threshold=50):
    for c in center_list:
        if abs(center[0] - c[0]) <= threshold and abs(center[1] - c[1]) <= threshold:
            return True
    return False

def process_video(cap, model, working_zone, conf_threshold, iou_threshold, batch_size, video_start_time, log_dir):
    fps, total_frames = get_video_info(cap)
    pbar = tqdm(total=total_frames, desc="🔍 Video işleniyor", unit="frame")

    frame_count = 0
    frames_batch = []
    three_sec_chunks = []
    fifteen_sec_chunks = []
    sixty_sec_chunks = []
    three_sec_log_json = []
    fifteen_sec_log_json = []
    sixty_sec_log_json = []
    processed_frames = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # NORMAL_SPEED kontrolü
            if not NORMAL_SPEED and frame_count % int(fps) != 0:
                frame_count += 1
                pbar.update(1)
                continue

            frames_batch.append(frame)

            if len(frames_batch) == 3:
                results = model(frames_batch, classes=[0], conf=conf_threshold, iou=iou_threshold, verbose=False)

                for idx, result in enumerate(results):
                    if DRAW_DETECTIONS:
                        draw_detections(frames_batch[idx], result.boxes, working_zone)

                    if SHOW_VIDEO:
                        if not show_frame(frames_batch[idx]):
                            break

                centers_f1 = boxes_to_centers(results[0].boxes, working_zone)
                centers_f2 = boxes_to_centers(results[1].boxes, working_zone)
                centers_f3 = boxes_to_centers(results[2].boxes, working_zone)

                confirmed_centers = []

                for center in centers_f2:
                    if match_center(center, centers_f1) or match_center(center, centers_f3):
                        confirmed_centers.append(center)

                for center in centers_f1:
                    if match_center(center, centers_f3) and not match_center(center, centers_f2):
                        confirmed_centers.append(center)

                inside_count = len(confirmed_centers)

                start_sec = (frame_count // int(fps)) - 2
                end_sec = start_sec + 3
                log_chunk(three_sec_log_json, start_sec, end_sec, inside_count, video_start_time, degisim_var=None)
                three_sec_chunks.append((start_sec, end_sec, inside_count))

                frames_batch = []
                processed_frames += 3

                while len(three_sec_chunks) >= 5:
                    candidate_chunks = three_sec_chunks[:5]
                    counts = [c[2] for c in candidate_chunks]
                    main_count = statistics.mode(counts)
                    sapma = sum(1 for count in counts if count != main_count)

                    degisim_var = counts[-1] != main_count

                    if sapma <= 1:
                        start_sec_15 = candidate_chunks[0][0]
                        end_sec_15 = candidate_chunks[-1][1]
                        log_chunk(fifteen_sec_log_json, start_sec_15, end_sec_15, main_count, video_start_time, degisim_var=degisim_var)
                        fifteen_sec_chunks.append((start_sec_15, end_sec_15, main_count, degisim_var))

                    three_sec_chunks = three_sec_chunks[5:]

                while len(fifteen_sec_chunks) >= 4:
                    last_four = fifteen_sec_chunks[:4]
                    counts_15 = [c[2] for c in last_four]

                    if len(set(counts_15)) == 1:
                        start_sec_60 = last_four[0][0]
                        end_sec_60 = last_four[-1][1]
                        kisi_sayisi = counts_15[0]
                        log_chunk(sixty_sec_log_json, start_sec_60, end_sec_60, kisi_sayisi, video_start_time, degisim_var=None)
                        sixty_sec_chunks.append((start_sec_60, end_sec_60, kisi_sayisi))

                    fifteen_sec_chunks = fifteen_sec_chunks[4:]

            frame_count += 1
            pbar.update(1)

    finally:
        cap.release()
        if SHOW_VIDEO:
            close_window()
        pbar.close()
        save_json(os.path.join(log_dir, "3_sec_chunks.json"), three_sec_log_json)
        save_json(os.path.join(log_dir, "15_sec_chunks.json"), fifteen_sec_log_json)
        save_json(os.path.join(log_dir, "60_sec_chunks.json"), sixty_sec_log_json)

    return processed_frames
