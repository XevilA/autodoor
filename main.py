import RPi.GPIO as GPIO
import time
import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

# GPIO Setup
DIR = 11       # Direction pin
STEP = 13      # Step pin
CW = 1         # Clockwise
CCW = 0        # Counter-Clockwise

GPIO.setmode(GPIO.BOARD)
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(STEP, GPIO.OUT)

# Ultrasonic Sensor Pins
GPIO_TRIGGER = 16
GPIO_ECHO = 18
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

# Camera Configuration
picam2 = Picamera2()
picam2.preview_configuration.main.size = (960, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# YOLO Model Initialization
model = YOLO('yolov8n.pt')
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# State Variables
door_open = False
count = 0

def distance():
    """วัดระยะทางด้วยอัลตราโซนิกเซ็นเซอร์พร้อมจัดการเวลา timeout"""
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)

    timeout = 0.1  # 100 ms
    start_time = time.time()

    # รอจนกว่า ECHO จะเริ่ม HIGH
    while GPIO.input(GPIO_ECHO) == 0:
        if time.time() - start_time > timeout:
            return -1  # Timeout
    echo_start = time.time()

    # รอจนกว่า ECHO จะกลับมา LOW
    while GPIO.input(GPIO_ECHO) == 1:
        if time.time() - echo_start > timeout:
            return -1  # Timeout
    echo_end = time.time()

    # คำนวณระยะทาง
    duration = echo_end - echo_start
    distance_cm = (duration * 34300) / 2
    return distance_cm

try:
    while True:
        im = picam2.capture_array()
        count += 1
        
        # ประมวลผลภาพทุก 3 เฟรมเพื่อลดภาระการประมวลผล
        if count % 3 != 0:
            continue
        
        im = cv2.flip(im, -1)  # หมุนภาพ 180 องศา
        results = model.predict(im)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        
        People_LIST = []
        
        # ตรวจหาวัตถุในเฟรม
        for _, row in px.iterrows():
            x1, y1, x2, y2, _, label = row
            label = int(label)
            obj_name = class_list[label]
            
            if obj_name == 'person':
                People_LIST.append((int(x1), int(y1), int(x2), int(y2)))
        
        People_Count = len(People_LIST)
        
        # วาด Bounding Box และข้อมูล
        for person in People_LIST:
            x1, y1, x2, y2 = person
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cvzone.putTextRect(im, 'Person', (x1, y1), scale=1, thickness=1)
        
        # แสดงจำนวนคน
        cvzone.putTextRect(im, f'People: {People_Count}', (200, 50),
                          scale=2, thickness=3, colorB=(0, 255, 0))
        
        # ควบคุมการเปิด-ปิดประตู
        if People_Count > 0 and not door_open:
            print("เปิดประตู")
            GPIO.output(DIR, CW)
            for _ in range(200):  # หมุนมอเตอร์ 200 สเต็ป
                GPIO.output(STEP, GPIO.HIGH)
                time.sleep(0.005)
                GPIO.output(STEP, GPIO.LOW)
                time.sleep(0.005)
            door_open = True
        
        elif door_open:
            dist = distance()
            print(f"ระยะทางวัดได้: {dist} ซม.")
            
            if dist == -1:
                print("ข้อผิดพลาดในการวัดระยะทาง")
            elif People_Count == 0 and dist >= 80:
                print("ปิดประตู")
                GPIO.output(DIR, CCW)
                for _ in range(200):
                    GPIO.output(STEP, GPIO.HIGH)
                    time.sleep(0.005)
                    GPIO.output(STEP, GPIO.LOW)
                    time.sleep(0.005)
                door_open = False
        
        cv2.imshow("Camera", im)
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("ทำความสะอาด GPIO")
    GPIO.cleanup()
    cv2.destroyAllWindows()
