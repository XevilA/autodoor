import RPi.GPIO as GPIO
import time
import threading
import tkinter as tk
from tkinter import ttk
from picamera2 import Picamera2
from ultralytics import YOLO

class SlidingDoorSystem:
    def __init__(self, root):
        self.setup_gpio()
        self.setup_camera()
        
        self.STEPS_PER_REV = 2300
        self.BELT_PITCH = 10
        self.PULLEY_TEETH = 80
        self.DOOR_WIDTH = 800
        self.SAFETY_DISTANCE = 50
        self.STEP_DELAY = 0.001
        
        self.mm_per_rev = self.BELT_PITCH * self.PULLEY_TEETH
        self.mm_per_step = self.mm_per_rev / self.STEPS_PER_REV
        self.total_steps = int(self.DOOR_WIDTH / self.mm_per_step)
        
        self.root = root
        self.setup_gui()
        
        self.door_position = 0.0
        self.is_running = True
        self.safety_triggered = False
        self.detection_active = True
        self.manual_mode = False
        
        threading.Thread(target=self.door_control_loop, daemon=True).start()
        threading.Thread(target=self.capture_loop, daemon=True).start()

    def setup_gpio(self):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        self.STEP_PIN = 13
        self.DIR_PIN = 11
        GPIO.setup(self.DIR_PIN, GPIO.OUT)
        GPIO.setup(self.STEP_PIN, GPIO.OUT)
        self.TRIG_PIN = 16
        self.ECHO_PIN = 18
        GPIO.setup(self.TRIG_PIN, GPIO.OUT)
        GPIO.setup(self.ECHO_PIN, GPIO.IN)

    def setup_camera(self):
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            raw={"size": (1536, 864)}
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(2)
        self.model = YOLO('yolov8n.pt')

    def setup_gui(self):
        self.root.title("Smart Sliding Door System")
        self.root.geometry("600x400")
        
        status_frame = ttk.LabelFrame(self.root, text="System Status")
        status_frame.pack(pady=10, padx=10, fill=tk.X)
        
        self.door_label = ttk.Label(status_frame, text="Door Position: 0.0 mm")
        self.door_label.pack(side=tk.LEFT, padx=5)
        
        self.safety_label = ttk.Label(status_frame, text="Safety: OK")
        self.safety_label.pack(side=tk.LEFT, padx=5)
        
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10, fill=tk.X)
        
        self.progress = ttk.Progressbar(control_frame, 
                                        orient=tk.HORIZONTAL,
                                        length=400,
                                        maximum=self.DOOR_WIDTH)
        self.progress.pack(pady=5)
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=5)
        
        ttk.Button(btn_frame, text="Emergency Stop", command=self.emergency_stop).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Toggle Detection", command=self.toggle_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Open", command=lambda: self.move_door(self.DOOR_WIDTH)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", command=lambda: self.move_door(0)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Toggle Manual Mode", command=self.toggle_manual_mode).pack(side=tk.LEFT, padx=5)

    def measure_distance(self):
        GPIO.output(self.TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIG_PIN, False)
        
        timeout = time.time() + 0.04
        start = end = time.time()
        
        while GPIO.input(self.ECHO_PIN) == 0 and time.time() < timeout:
            start = time.time()
        while GPIO.input(self.ECHO_PIN) == 1 and time.time() < timeout:
            end = time.time()
        
        return (end - start) * 17150

    def move_door(self, target_mm):
        if self.safety_triggered:
            return
        
        steps = int(abs(target_mm - self.door_position) / self.mm_per_step)
        if steps == 0:
            return
        
        direction = GPIO.HIGH if target_mm > self.door_position else GPIO.LOW
        GPIO.output(self.DIR_PIN, direction)
        
        for _ in range(steps):
            if self.safety_triggered:
                break
            GPIO.output(self.STEP_PIN, GPIO.HIGH)
            time.sleep(self.STEP_DELAY)
            GPIO.output(self.STEP_PIN, GPIO.LOW)
            time.sleep(self.STEP_DELAY)
            self.door_position += self.mm_per_step * (1 if direction == GPIO.HIGH else -1)
            self.update_gui()
        
        if target_mm == self.DOOR_WIDTH:
            time.sleep(5)  
            self.move_door(0)  
    def door_control_loop(self):
        while self.is_running:
            if self.detection_active and not self.manual_mode:
                current_distance = self.measure_distance()
                self.safety_triggered = current_distance < self.SAFETY_DISTANCE
                
                if self.safety_triggered and self.door_position > 0:
                    self.move_door(0)
                    self.safety_label.config(text="Safety: OBSTACLE DETECTED!")
                else:
                    self.safety_label.config(text="Safety: OK")
            time.sleep(0.1)

    def capture_loop(self):
        while self.is_running:
            if self.detection_active and not self.manual_mode:
                frame = self.camera.capture_array()
                results = self.model.predict(frame, classes=0, conf=0.65, verbose=False)
                
                if len(results[0].boxes) > 0 and not self.safety_triggered:
                    self.move_door(self.DOOR_WIDTH)
            time.sleep(1)

    def update_gui(self):
        self.progress['value'] = self.door_position
        self.door_label.config(text=f"Door Position: {self.door_position:.1f} mm")
        self.root.update()

    def emergency_stop(self):
        self.safety_triggered = True
        self.move_door(0)
        self.detection_active = False

    def toggle_detection(self):
        self.detection_active = not self.detection_active
        status = "ACTIVE" if self.detection_active else "DISABLED"
        self.safety_label.config(text=f"Detection: {status}")

    def toggle_manual_mode(self):
        self.manual_mode = not self.manual_mode

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
