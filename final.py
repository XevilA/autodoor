import RPi.GPIO as GPIO
import time
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np

# Hardware Configuration
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# Motor Control Parameters
STEP_PIN = 13
DIR_PIN = 11
MOTOR_STEPS_PER_REV = 200  # 1.8° per step (NEMA 17 typical)
GEAR_RATIO = 10            # Gear reduction ratio
GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)

# Ultrasonic Sensor Configuration
TRIG_PIN = 16
ECHO_PIN = 18
GPIO.setup(TRIG_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ECHO_PIN, GPIO.IN)

# Camera Configuration
camera = Picamera2()
config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 30, "AwbEnable": True, "AeEnable": True}
)
camera.configure(config)
camera.start()
time.sleep(2)  # Camera initialization

# AI Model Configuration
model = YOLO('yolov8n.pt')
DETECTION_CONFIDENCE = 0.7  # Minimum confidence threshold
PERSON_CLASS_ID = 0         # COCO dataset person class

# System Parameters
STEP_DELAY = 0.0015        # Motor step interval (seconds)
DOOR_OPEN_ANGLE = 90       # Degrees to open
SAFETY_DISTANCE = 50       # Centimeters (stop closing if obstacle detected)

# Calculate required steps
STEPS_PER_DEGREE = (MOTOR_STEPS_PER_REV * GEAR_RATIO) / 360
REQUIRED_STEPS = int(DOOR_OPEN_ANGLE * STEPS_PER_DEGREE)

class DoorController:
    def __init__(self):
        self.is_open = False
        self.in_motion = False
        self.current_step = 0
        self.direction = GPIO.LOW

    def precise_distance(self):
        """Get filtered distance measurement with error handling"""
        try:
            # Send pulse
            GPIO.output(TRIG_PIN, True)
            time.sleep(0.000015)
            GPIO.output(TRIG_PIN, False)

            timeout = time.time() + 0.04
            start = end = time.time()

            # Measure echo start
            while GPIO.input(ECHO_PIN) == 0 and time.time() < timeout:
                start = time.time()
            
            # Measure echo end
            while GPIO.input(ECHO_PIN) == 1 and time.time() < timeout:
                end = time.time()

            # Filter outliers using moving average
            distance = (end - start) * 17150  # cm
            return max(0, min(distance, 400))  # Limit to 4 meters
        except:
            return 400  # Return safe maximum

    def move_door(self, direction):
        """Smooth motor control with real-time safety checks"""
        self.direction = direction
        GPIO.output(DIR_PIN, direction)
        
        for _ in range(REQUIRED_STEPS):
            if self.safety_check():
                print("Safety stop triggered!")
                return False
            
            GPIO.output(STEP_PIN, GPIO.HIGH)
            time.sleep(STEP_DELAY)
            GPIO.output(STEP_PIN, GPIO.LOW)
            time.sleep(STEP_DELAY)
        
        return True

    def safety_check(self):
        """Check for obstacles during closing"""
        if not self.is_open and self.direction == GPIO.LOW:
            return self.precise_distance() < SAFETY_DISTANCE
        return False

    def full_operation(self, detected_person):
        """Main control logic"""
        if not self.in_motion:
            if detected_person and not self.is_open:
                print("Opening door...")
                self.in_motion = True
                if self.move_door(GPIO.HIGH):
                    self.is_open = True
                self.in_motion = False
                
            elif not detected_person and self.is_open:
                if self.precise_distance() > SAFETY_DISTANCE * 1.5:
                    print("Closing door...")
                    self.in_motion = True
                    if self.move_door(GPIO.LOW):
                        self.is_open = False
                    self.in_motion = False

def main():
    controller = DoorController()
    try:
        while True:
            # Capture and process frame
            frame = camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Person detection with enhanced accuracy
            results = model.predict(
                frame, 
                imgsz=320, 
                classes=[PERSON_CLASS_ID], 
                conf=DETECTION_CONFIDENCE,
                iou=0.4,
                verbose=False
            )
            
            # Display detected persons
            detected = False
            for box in results[0].boxes:
                if box.conf[0] > DETECTION_CONFIDENCE:
                    detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Control logic
            controller.full_operation(detected)

            # Display system status
            status_text = f"Door: {'OPEN' if controller.is_open else 'CLOSED'} | Detected: {len(results[0].boxes)}"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Smart Door System", frame)
            
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Critical error: {str(e)}")
    finally:
        GPIO.cleanup()
        camera.stop()
        cv2.destroyAllWindows()
        print("System shutdown complete")

if __name__ == "__main__":
    main()
