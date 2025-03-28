import RPi.GPIO as GPIO
import time
import threading
import tkinter as tk
from tkinter import ttk
from picamera2 import Picamera2
from ultralytics import YOLO

class SlidingDoorSystem:
    def __init__(self, root):
        # Hardware Configuration
        self.setup_gpio()
        self.setup_camera()
        
        # GUI Setup
        self.root = root
        self.setup_gui()
        
        # System Parameters
        self.STEPS_PER_REV = 2300    
        self.BELT_PITCH = 10          
        self.PULLEY_TEETH = 80       
        self.DOOR_WIDTH = 800         
        

        self.mm_per_rev = self.BELT_PITCH * self.PULLEY_TEETH  
        self.mm_per_step = self.mm_per_rev / self.STEPS_PER_REV
        self.total_steps = int(self.DOOR_WIDTH / self.mm_per_step)  
        
        # System State
        self.door_position = 0  # 0-800 mm
        self.is_running = True
        self.safety_stop = False
        
        # Start Threads
        threading.Thread(target=self.door_control_loop, daemon=True).start()
        threading.Thread(target=self.capture_loop, daemon=True).start()

    def setup_gpio(self):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        self.STEP_PIN = 13
        self.DIR_PIN = 11
        GPIO.setup(self.DIR_PIN, GPIO.OUT)
        GPIO.setup(self.STEP_PIN, GPIO.OUT)

    def setup_camera(self):
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_preview_configuration(main={"size": (640, 480)}))
        self.camera.start()
        self.model = YOLO('yolov8n.pt')

    def setup_gui(self):
        self.root.title("Sliding Door Control")
        status_frame = ttk.LabelFrame(self.root, text="System Status")
        status_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.door_label = ttk.Label(status_frame, text="Location: 0 mm")
        self.door_label.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=400, maximum=self.DOOR_WIDTH)
        self.progress.pack(padx=10, pady=5)
        
        ttk.Button(self.root, text="Door Opening", command=lambda: self.move_door(self.DOOR_WIDTH)).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.root, text="Door Closing", command=lambda: self.move_door(0)).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.root, text="EMER", command=self.emergency_stop).pack(side=tk.RIGHT, padx=5)

    def move_door(self, target_mm):
        def _move():
            steps = int(abs(target_mm - self.door_position) / self.mm_per_step)
            direction = GPIO.HIGH if target_mm > self.door_position else GPIO.LOW
            
            GPIO.output(self.DIR_PIN, direction)
            for _ in range(steps):
                if self.safety_stop: 
                    break
                GPIO.output(self.STEP_PIN, GPIO.HIGH)
                time.sleep(0.001)  # ปรับความเร็วที่นี่
                GPIO.output(self.STEP_PIN, GPIO.LOW)
                time.sleep(0.001)
                self.door_position += self.mm_per_step * (1 if direction == GPIO.HIGH else -1)
                self.update_gui()
        
        threading.Thread(target=_move, daemon=True).start()

    def update_gui(self):
        self.progress['value'] = self.door_position
        self.door_label.config(text=f"location: {self.door_position:.1f} mm")
        self.root.update()

    def door_control_loop(self):
        while self.is_running:
            frame = self.camera.capture_array()
            results = self.model.predict(frame, classes=0, conf=0.65)
            
            if any(box.conf[0] > 0.65 for box in results[0].boxes):
                if self.door_position < self.DOOR_WIDTH:
                    self.move_door(self.DOOR_WIDTH)  
            else:
                if self.door_position > 0:
                    self.move_door(0)  
            time.sleep(0.5)

    def emergency_stop(self):
        self.safety_stop = True
        self.door_position = 0
        GPIO.output(self.DIR_PIN, GPIO.LOW)
        self.update_gui()

    def shutdown(self):
        self.is_running = False
        self.camera.stop()
        GPIO.cleanup()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SlidingDoorSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.shutdown)
    root.mainloop()
