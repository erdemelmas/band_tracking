import cv2
import numpy as np

def draw_detections(frame, boxes, working_zone):
    # Çalışma alanını çiz (Mavi)
    pts = np.array(working_zone, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        center = (center_x, center_y)

        # Bu kişi çalışma alanı içinde mi?
        inside = cv2.pointPolygonTest(pts, center, False) >= 0

        if inside:
            box_color = (0, 255, 0)  # Yeşil: İçeride
        else:
            box_color = (0, 0, 255)  # Kırmızı: Dışarıda

        # Kutu çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        # Merkeze küçük bir daire çiz
        cv2.circle(frame, center, 4, box_color, -1)

def show_frame(frame, window_name="Band Tracking Video"):
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False  # 'q' basıldıysa videodan çıkmak istiyor
    return True

def close_window():
    cv2.destroyAllWindows()
