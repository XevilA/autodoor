import RPi.GPIO as GPIO
import time
import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

# Setup GPIO for Stepper Motor
DIR = 11
STEP = 13
CW = 1
CCW = 0

GPIO.setmode(GPIO.BOARD)
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(STEP, GPIO.OUT)
GPIO.output(DIR, CW)

# Ultrasonic Sensor Pins
GPIO_TRIGGER = 16
GPIO_ECHO = 18
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

# Camera Setup
picam2 = Picamera2()
picam2.preview_configuration.main.size = (960, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# YOLO Model
model = YOLO('yolov8n.pt')
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# State Variables
door_open = False
motor_steps = 0
motor_direction = CW
CONFIDENCE_THRESHOLD = 0.5
STEP_DELAY = 0.005
REQUIRED_STEPS = 200
DISTANCE_THRESHOLD = 80  # cm

def distance():
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    
    timeout = 0.04  # 40ms timeout (~6.8m max distance)
    start_time = time.time()
    
    # Wait for echo start
    while GPIO.input(GPIO_ECHO) == 0:
        if time.time() - start_time > timeout:
            return -1
    StartTime = time.time()
    
    # Wait for echo end
    start_time = time.time()
    while GPIO.input(GPIO_ECHO) == 1:
        if time.time() - start_time > timeout:
            return -1
    StopTime = time.time()
    
    TimeElapsed = StopTime - StartTime
    distance_cm = (TimeElapsed * 34300) / 2
    return distance_cm

try:
    while True:
        im = picam2.capture_array()
        im = cv2.flip(im, -1)  # Adjust if camera is rotated
        
        # YOLO Detection every 3rd frame
        results = model.predict(im)
        boxes = results[0].boxes
        conf_mask = boxes.conf.cpu().numpy() > CONFIDENCE_THRESHOLD
        filtered_boxes = boxes.data[conf_mask]
        px = pd.DataFrame(filtered_boxes.cpu().numpy(), columns=['x1', 'y1', 'x2', 'y3', 'conf', 'class'])
        
        people_list = []
        for index, row in px.iterrows():
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            label_index = int(row[5])
            obj_name = class_list[label_index]
            
            if obj_name == 'person':
                people_list.append((x1, y1, x2, y2))
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cvzone.putTextRect(im, 'Person', (x1, y1), scale=1, thickness=1)
        
        people_count = len(people_list)
        cvzone.putTextRect(im, f'People: {people_count}', (50, 50), scale=2, thickness=3)
        
        # Door Control Logic
        if people_count > 0 and not door_open:
            # Start opening door
            motor_direction = CW
            motor_steps = REQUIRED_STEPS
            door_open = True  # Assume door will be open after steps
            print("Opening door...")
        
        # Handle motor movement non-blocking
        if motor_steps > 0:
            GPIO.output(DIR, motor_direction)
            GPIO.output(STEP, GPIO.HIGH)
            time.sleep(STEP_DELAY)
            GPIO.output(STEP, GPIO.LOW)
            time.sleep(STEP_DELAY)
            motor_steps -= 1
            if motor_steps == 0:
                print(f"Door {'opened' if motor_direction == CW else 'closed'}.")
        
        # Check distance if door is open and no people
        if door_open and people_count == 0:
            current_distance = distance()
            if current_distance >= DISTANCE_THRESHOLD or current_distance == -1:
                # Start closing door
                motor_direction = CCW
                motor_steps = REQUIRED_STEPS
                door_open = False
                print("Closing door...")
        
        cv2.imshow("Camera", im)
        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")
finally:
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()
