import cv2
import numpy as np
import uuid
from datetime import datetime

# Функция для изменения размера кадра
def resize_frame(frame, width=None, height=None):
    (h, w) = frame.shape[:2]
    if width and height:
        resized = cv2.resize(frame, (width, height))
    else:
        if width is None:
            ratio = height / float(h)
            dimension = (int(w * ratio), height)
        else:
            ratio = width / float(w)
            dimension = (width, int(h * ratio))
        resized = cv2.resize(frame, dimension)
    return resized

# Функция для создания маски статичных световых пятен
def create_static_light_mask(frame, threshold=230):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, light_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return light_mask

# Функция для отслеживания и обновления дефектов
def detect_dents(frame, polygon_points, light_mask, detected_defects):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 255)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(light_mask))
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)

    _, thresh = cv2.threshold(gradient, 35, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    new_defects = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        if radius > 3:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.75 < aspect_ratio < 1.25:
                new_defect = True

                # Проверка, не был ли дефект уже зафиксирован
                for defect_id, defect_data in detected_defects.items():
                    prev_x, prev_y, prev_w, prev_h, last_seen, classification = defect_data
                    # Сравниваем положение и радиус, если схоже - продолжаем отслеживать
                    if abs(x - prev_x) < 50 and abs(y - prev_y) < 50:
                        # Обновляем время последнего обнаружения
                        detected_defects[defect_id] = (x, y, w, h, datetime.now(), classification)
                        new_defect = False
                        cv2.circle(frame, center, radius, (255, 0, 0), 2)
                        cv2.putText(frame, f"{classification} ID: {defect_id}", (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        break

                # Добавляем новый дефект, если он ранее не был зарегистрирован
                if new_defect:
                    defect_id = str(uuid.uuid4())[:8]
                    classification = "Dent"  # Здесь можно добавить более сложную классификацию
                    detected_defects[defect_id] = (x, y, w, h, datetime.now(), classification)
                    cv2.circle(frame, center, radius, (0, 0, 255), 2)
                    cv2.putText(frame, f"New {classification} ID: {defect_id}", (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame, detected_defects

cap = cv2.VideoCapture('/home/robert/def_scan/photo/video/point.avi')
desired_width = 1280
desired_height = 720

if not cap.isOpened():
    print("Ошибка: не удалось открыть видеопоток!")
    exit()

ret, initial_frame = cap.read()
if not ret:
    print("Ошибка: не удалось считать первый кадр!")
    cap.release()
    exit()

initial_resized_frame = resize_frame(initial_frame, width=desired_width, height=desired_height)
light_mask = create_static_light_mask(initial_resized_frame)

polygon_points = np.array([[0, 360], [700, 70], [1150, 90], [1100, 700], [0, 700]])

detected_defects = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось считать кадр!")
        break

    resized_frame = resize_frame(frame, width=desired_width, height=desired_height)
    output_frame, detected_defects = detect_dents(resized_frame, polygon_points, light_mask, detected_defects)

    # Рисуем многоугольник зоны интереса на кадре
    cv2.polylines(output_frame, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Detected Dents - Video Stream", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
