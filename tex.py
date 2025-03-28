import RPi.GPIO as GPIO
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from picamera2 import Picamera2
from ultralytics import YOLO
import logging

class SlidingDoorSystem:
    def __init__(self, root):
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        try:
            self.setup_gpio()
            self.setup_camera()
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            messagebox.showerror("Initialization Error", str(e))
            return
        
        # Configuration constants (adjustable)
        self.STEPS_PER_REV = 2300        
        self.BELT_PITCH = 10             
        self.PULLEY_TEETH = 80           
        self.DOOR_WIDTH = 800  # Updated to 80cm (800mm)
        self.SAFETY_DISTANCE = 60        
        self.STEP_DELAY = 0.001          
        self.DETECTION_CONFIDENCE = 0.65
        
        # Calculated parameters
        self.mm_per_rev = self.BELT_PITCH * self.PULLEY_TEETH
        self.mm_per_step = self.mm_per_rev / self.STEPS_PER_REV
        self.total_steps = int(self.DOOR_WIDTH / self.mm_per_step)
        
        self.root = root
        self.setup_gui()
        
        # State variables
        self.door_position = 0.0         
        self.is_running = True
        self.manual_mode = False
        
        # Start control threads
        try:
            threading.Thread(target=self.door_control_loop, daemon=True).start()
            threading.Thread(target=self.capture_loop, daemon=True).start()
        except Exception as e:
            self.logger.error(f"Thread startup error: {e}")
            messagebox.showerror("Thread Error", "Could not start background threads")

    def setup_gpio(self):
        """Set up GPIO pins with error handling"""
        try:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
            
            # Motor control pins
            self.STEP_PIN = 13
            self.DIR_PIN = 11
            GPIO.setup(self.DIR_PIN, GPIO.OUT)
            GPIO.setup(self.STEP_PIN, GPIO.OUT)
            
            # Ultrasonic sensor pins
            self.TRIG_PIN = 16
            self.ECHO_PIN = 18
            GPIO.setup(self.TRIG_PIN, GPIO.OUT)
            GPIO.setup(self.ECHO_PIN, GPIO.IN)
            
            self.logger.info("GPIO setup completed successfully")
        except Exception as e:
            self.logger.error(f"GPIO setup failed: {e}")
            raise

    def setup_camera(self):
        """Initialize camera with comprehensive error handling"""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            
            # Wait for camera to initialize
            time.sleep(2)  
            
            # Load YOLO model
            self.model = YOLO('yolov8n.pt')
            
            self.logger.info("Camera and YOLO model initialized successfully")
        except Exception as e:
            self.logger.error(f"Camera setup failed: {e}")
            raise

    def setup_gui(self):
        """Create GUI with enhanced layout and error handling"""
        try:
            self.root.title("Smart Sliding Door System")
            self.root.geometry("600x400")
            
            # Door position progress bar
            ttk.Label(self.root, text="Door Position", font=('Helvetica', 12)).pack(pady=5)
            self.progress = ttk.Progressbar(
                self.root, 
                orient=tk.HORIZONTAL, 
                length=400, 
                maximum=self.DOOR_WIDTH
            )
            self.progress.pack(pady=10)
            
            # Control buttons
            control_frame = ttk.Frame(self.root)
            control_frame.pack(pady=10)
            
            ttk.Button(control_frame, text="Manual Open", 
                       command=lambda: self.move_door(self.DOOR_WIDTH)).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text="Manual Close", 
                       command=lambda: self.move_door(0)).pack(side=tk.LEFT, padx=5)
            
            # Status label
            self.status_var = tk.StringVar(value="System Ready")
            ttk.Label(self.root, textvariable=self.status_var, font=('Helvetica', 10)).pack(pady=5)
        
        except Exception as e:
            self.logger.error(f"GUI setup failed: {e}")
            messagebox.showerror("GUI Error", "Could not create user interface")

    def measure_distance(self):
        """Measure distance using ultrasonic sensor with improved reliability"""
        try:
            # Trigger ultrasonic pulse
            GPIO.output(self.TRIG_PIN, False)
            time.sleep(0.1)
            GPIO.output(self.TRIG_PIN, True)
            time.sleep(0.00001)
            GPIO.output(self.TRIG_PIN, False)
            
            # Wait for echo start
            timeout = time.time() + 1  # 1-second timeout
            while GPIO.input(self.ECHO_PIN) == 0:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return float('inf')
            
            # Wait for echo end
            while GPIO.input(self.ECHO_PIN) == 1:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return float('inf')
            
            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 17150  # Speed of sound conversion
            
            return distance
        
        except Exception as e:
            self.logger.error(f"Distance measurement error: {e}")
            return float('inf')

    def move_door(self, target_mm):
        """Move door with improved precision and logging"""
        try:
            steps = int(abs(target_mm - self.door_position) / self.mm_per_step)
            if steps == 0:
                return

            direction = GPIO.HIGH if target_mm > self.door_position else GPIO.LOW
            GPIO.output(self.DIR_PIN, direction)

            for _ in range(steps):
                GPIO.output(self.STEP_PIN, GPIO.HIGH)
                time.sleep(self.STEP_DELAY)
                GPIO.output(self.STEP_PIN, GPIO.LOW)
                time.sleep(self.STEP_DELAY)
                
                # Update door position
                self.door_position += self.mm_per_step * (1 if direction == GPIO.HIGH else -1)
                self.update_gui()
            
            self.logger.info(f"Door moved to {target_mm}mm")
            self.status_var.set(f"Door at {self.door_position:.1f}mm")
        
        except Exception as e:
            self.logger.error(f"Door movement error: {e}")
            self.status_var.set("Movement Error")

    def door_control_loop(self):
        """Background thread for door control"""
        while self.is_running:
            time.sleep(1)
    
    def capture_loop(self):
        """Continuous object detection and door control"""
        while self.is_running:
            try:
                # Capture frame
                frame = self.camera.capture_array()
                
                # Detect people (class 0 in COCO dataset)
                results = self.model.predict(
                    frame, 
                    classes=0,  # People detection 
                    conf=self.DETECTION_CONFIDENCE, 
                    verbose=False
                )
                
                # If people detected, open door
                if len(results[0].boxes) > 0:
                    self.status_var.set("Person Detected - Opening Door")
                    self.move_door(self.DOOR_WIDTH)
                    time.sleep(4)
                    
                    # Wait until clear of obstruction
                    while self.measure_distance() < self.SAFETY_DISTANCE:
                        time.sleep(1)
                    
                    self.move_door(0)
                    self.status_var.set("Door Closed")
                
                time.sleep(1)
            
            except Exception as e:
                self.logger.error(f"Capture loop error: {e}")
                time.sleep(2)

    def update_gui(self):
        """Update GUI progress bar"""
        try:
            self.progress['value'] = self.door_position
            self.root.update()
        except Exception as e:
            self.logger.error(f"GUI update error: {e}")

    def shutdown(self):
        """Graceful system shutdown"""
        try:
            self.is_running = False
            self.camera.stop()
            GPIO.cleanup()
            self.logger.info("System shutdown initiated")
            self.root.destroy()
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SlidingDoorSystem(root)
        root.protocol("WM_DELETE_WINDOW", app.shutdown)
        root.mainloop()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
