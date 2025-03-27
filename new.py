import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np

class SmartDoorController:
    def __init__(self):
        # Hardware Configuration
        self.setup_gpio()
        self.setup_camera()
        self.setup_parameters()
        
        # Initialize Components
        self.model = YOLO('yolov8n.pt')
        self.current_angle = 0.0

    def setup_gpio(self):
        """Initialize GPIO settings"""
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        
        # Stepper Motor Pins
        self.STEP_PIN = 13
        self.DIR_PIN = 11
        GPIO.setup(self.STEP_PIN, GPIO.OUT)
        GPIO.setup(self.DIR_PIN, GPIO.OUT)
        
        # Ultrasonic Sensor Pins
        self.TRIG_PIN = 16
        self.ECHO_PIN = 18
        GPIO.setup(self.TRIG_PIN, GPIO.OUT)
        GPIO.setup(self.ECHO_PIN, GPIO.IN)

    def setup_camera(self):
        """Configure camera settings"""
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            raw={"size": (1536, 864)}
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(2)  # Camera warm-up

    def setup_parameters(self):
        """Adjustable system parameters"""
        # Motor Technical Specs
        self.STEP_ANGLE = 1.8          # Degrees per step (NEMA 17 = 1.8°)
        self.GEAR_RATIO = 5            # Gear reduction ratio
        self.STEP_DELAY = 0.002        # Seconds between steps
        
        # Door Settings
        self.MAX_ANGLE = 90.0         # Maximum opening angle (degrees)
        self.SAFETY_DISTANCE = 50.0    # Obstacle detection distance (cm)
        
        # Calculated Values
        self.STEPS_PER_DEGREE = (360 / self.STEP_ANGLE) * self.GEAR_RATIO
        self.TOTAL_STEPS = int(self.MAX_ANGLE * self.STEPS_PER_DEGREE)

    def calculate_door_angle(self, steps):
        """Convert steps to door angle"""
        return steps / self.STEPS_PER_DEGREE

    def get_filtered_distance(self):
        """Get reliable distance measurement"""
        distances = []
        for _ in range(3):
            distances.append(self.measure_distance())
            time.sleep(0.01)
        return np.median(distances)

    def measure_distance(self):
        """Single distance measurement"""
        GPIO.output(self.TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIG_PIN, False)

        timeout = time.time() + 0.04
        start = end = time.time()

        while GPIO.input(self.ECHO_PIN) == 0 and time.time() < timeout:
            start = time.time()
            
        while GPIO.input(self.ECHO_PIN) == 1 and time.time() < timeout:
            end = time.time()

        return (end - start) * 17150  # cm

    def detect_person(self):
        """Person detection with YOLO"""
        frame = self.camera.capture_array()
        results = self.model.predict(
            frame,
            classes=0,  # Person class only
            conf=0.65,  # Confidence threshold
            imgsz=320,
            verbose=False
        )
        return len(results[0].boxes) > 0

    def move_door(self, target_angle):
        """Precise angle control with safety check"""
        target_steps = int(target_angle * self.STEPS_PER_DEGREE)
        direction = GPIO.HIGH if target_angle > self.current_angle else GPIO.LOW
        GPIO.output(self.DIR_PIN, direction)

        for _ in range(abs(target_steps - int(self.current_angle * self.STEPS_PER_DEGREE))):
            if self.safety_check(direction):
                print("Safety stop triggered!")
                return False
            
            GPIO.output(self.STEP_PIN, GPIO.HIGH)
            time.sleep(self.STEP_DELAY)
            GPIO.output(self.STEP_PIN, GPIO.LOW)
            time.sleep(self.STEP_DELAY)
            
            self.current_angle += 1/self.STEPS_PER_DEGREE if direction == GPIO.HIGH else -1/self.STEPS_PER_DEGREE
        
        return True

    def safety_check(self, direction):
        """Check for obstacles during closing"""
        if direction == GPIO.LOW:  # Closing direction
            return self.get_filtered_distance() < self.SAFETY_DISTANCE
        return False

    def run(self):
        try:
            while True:
                # Control Logic
                target_angle = self.MAX_ANGLE if self.detect_person() else 0.0
                
                if not np.isclose(self.current_angle, target_angle, atol=0.5):
                    print(f"Moving to {target_angle:.1f}°")
                    self.move_door(target_angle)
                
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
        except Exception as e:
            print(f"System error: {str(e)}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Cleanup resources"""
        GPIO.output(self.DIR_PIN, GPIO.LOW)
        GPIO.output(self.STEP_PIN, GPIO.LOW)
        GPIO.cleanup()
        self.camera.stop()
        print("System shutdown complete")

if __name__ == "__main__":
    door_system = SmartDoorController()
    door_system.run()
