import RPi.GPIO as GPIO
import time
import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

# กำหนดขา GPIO สำหรับควบคุมมอเตอร์สเต็ปเปอร์
DIR = 11   # ขาสำหรับกำหนดทิศทาง
STEP = 13  # ขาสำหรับสั่งให้มอเตอร์หมุน
CW = 1     # หมุนตามเข็มนาฬิกา
CCW = 0    # หมุนทวนเข็มนาฬิกา
SPR = 200  # จำนวนสเต็ปต่อการหมุน 1 รอบ (ขึ้นอยู่กับมอเตอร์ของคุณ)

# กำหนดขา GPIO สำหรับเซ็นเซอร์อัลตราโซนิก
GPIO_TRIGGER = 16
GPIO_ECHO = 18

# ตั้งค่า GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(STEP, GPIO.OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

# ฟังก์ชันวัดระยะทางจากเซ็นเซอร์อัลตราโซนิก
def distance():
    # ส่งสัญญาณ Trigger
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)

    StartTime = time.time()
    StopTime = time.time()

    # รอรับสัญญาณ Echo
    timeout = StartTime + 0.1  # กำหนด timeout ที่ 0.1 วินาที
    while GPIO.input(GPIO_ECHO) == 0 and StartTime < timeout:
        StartTime = time.time()

    if StartTime >= timeout:
        return None  # หากไม่มีสัญญาณ Echo กลับมา

    timeout = StartTime + 0.1
    while GPIO.input(GPIO_ECHO) == 1 and StopTime < timeout:
        StopTime = time.time()

    if StopTime >= timeout:
        return None  # หากสัญญาณ Echo ยาวเกินไป

    # คำนวณระยะทาง
    TimeElapsed = StopTime - StartTime
    distance = (TimeElapsed * 34300) / 2
    return distance

# ฟังก์ชันควบคุมการหมุนของมอเตอร์สเต็ปเปอร์
def move_stepper(steps, direction, step_delay=0.005):
    GPIO.output(DIR, direction)
    for _ in range(steps):
        GPIO.output(STEP, GPIO.HIGH)
        time.sleep(step_delay)
        GPIO.output(STEP, GPIO.LOW)
        time.sleep(step_delay)

# ตั้งค่า Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (960, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# โหลดโมเดล YOLO
model = YOLO('yolov8n.pt')
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

try:
    while True:
        # จับภาพจากกล้อง
        im = picam2.capture_array()
        im = cv2.flip(im, -1)

        # ตรวจจับวัตถุด้วย YOLO
        results = model.predict(im)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        People_LIST = []

        for index, row in px.iterrows():
            x1, y1, x2, y2, _, Label_Index = map(int, row)
            Object_Name = class_list[Label_Index]

            if Object_Name == 'person':
                People_LIST.append((x1, y1, x2, y2, Object_Name))
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cvzone.putTextRect(im, 'People', (x1, y1), 1, 1)

        People_Count = len(People_LIST)
        cvzone.putTextRect(im, f'People: {People_Count}', (200, 50),
                           colorT=(0, 0, 0), colorB=(0, 255, 0), colorR=(0, 255, 0),
                           scale=2, thickness=3, border=5, font=cv2.FONT_HERSHEY_COMPLEX_SMALL)

        if People_Count > 0:
            print(" --> Direction: CW")
            move_stepper(SPR, CW)

            dist = distance()
            if dist is not None:
                print(f" --> Distance = {dist:.2f} cm => Door Open")
                while dist < 80:
                    dist = distance()
                    if dist is None:
                        print(" --> Distance measurement failed")
                        break
                    print(f" --> Distance = {dist:.2f} cm => Door Open")

            print(" --> Direction: CCW => Door Close")
            move_stepper(SPR, CCW)

        cv2.imshow("Camera", im)

        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    print("Cleaning up GPIO and closing windows")
    GPIO.cleanup()
    cv2.destroyAllWindows()
