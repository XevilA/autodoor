import RPi.GPIO as GPIO
import time
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import cvzone

# GPIO Setup
GPIO.setmode(GPIO.BOARD)
DIR = 11
STEP = 13
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(STEP, GPIO.OUT)

# Ultrasonic Sensor
GPIO_TRIGGER = 16
GPIO_ECHO = 18
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

# Camera Setup
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# YOLO Model
model = YOLO('yolov8n.pt')

# System Parameters
STEP_DELAY = 0.002
TOTAL_STEPS = 400
DOOR_OPEN = False
MOTOR_BUSY = False
current_step = 0

def distance():
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    
    start_time = time.time()
    timeout = 0.04
    
    while GPIO.input(GPIO_ECHO) == 0:
        if time.time() - start_time > timeout:
            return 1000
    start = time.time()
    
    while GPIO.input(GPIO_ECHO) == 1:
        if time.time() - start_time > timeout:
            return 1000
    end = time.time()
    
    return (end - start) * 17150

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, -1)
        results = model.predict(frame, verbose=False)
        
        # People Detection
        people_detected = False
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if results[0].names[cls] == 'person' and box.conf[0] > 0.5:
                people_detected = True
                break

        # Door Control Logic
        if not MOTOR_BUSY:
            if people_detected and not DOOR_OPEN:
                # Start opening door
                GPIO.output(DIR, GPIO.HIGH)
                current_step = 0
                MOTOR_BUSY = True
                print("OPENING DOOR")
                
            elif not people_detected and DOOR_OPEN:
                # Check distance before closing
                dist = distance()
                if dist > 80 or dist < 0:
                    # Start closing door
                    GPIO.output(DIR, GPIO.LOW)
                    current_step = 0
                    MOTOR_BUSY = True
                    print("CLOSING DOOR")

        # Motor Control
        if MOTOR_BUSY:
            GPIO.output(STEP, GPIO.HIGH)
            time.sleep(STEP_DELAY)
            GPIO.output(STEP, GPIO.LOW)
            time.sleep(STEP_DELAY)
            
            current_step += 1
            if current_step >= TOTAL_STEPS:
                MOTOR_BUSY = False
                DOOR_OPEN = not DOOR_OPEN
                print(f"DOOR {'OPEN' if DOOR_OPEN else 'CLOSED'}")

        # Display
        cv2.putText(frame, f"DOOR: {'OPEN' if DOOR_OPEN else 'CLOSED'}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Smart Door", frame)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()
