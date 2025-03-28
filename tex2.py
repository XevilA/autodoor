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
        self.SAFETY_DISTANCE = 60        
        self.STEP_DELAY = 0.001          
        
        self.mm_per_rev = self.BELT_PITCH * self.PULLEY_TEETH
        self.mm_per_step = self.mm_per_rev / self.STEPS_PER_REV
        self.total_steps = int(self.DOOR_WIDTH / self.mm_per_step)
        
        self.root = root
        self.setup_gui()
        
        self.door_position = 0.0         
        self.is_running = True
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
        config = self.camera.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        self.camera.configure(config)
        self.camera.start()
        time.sleep(2)  
        self.model = YOLO('yolov8n.pt')

    def setup_gui(self):
        self.root.title("Smart Sliding Door System")
        self.root.geometry("600x400")
        
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=400, maximum=self.DOOR_WIDTH)
        self.progress.pack(pady=10)
        
        ttk.Button(self.root, text="Manual Open", command=lambda: self.move_door(self.DOOR_WIDTH)).pack()
        ttk.Button(self.root, text="Manual Close", command=lambda: self.move_door(0)).pack()
    
    def measure_distance(self):
        GPIO.output(self.TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIG_PIN, False)
        
        start, end = time.time(), time.time()
        while GPIO.input(self.ECHO_PIN) == 0:
            start = time.time()
        while GPIO.input(self.ECHO_PIN) == 1:
            end = time.time()
        
        return (end - start) * 17150  

    def move_door(self, target_mm):
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
            self.door_position += self.mm_per_step * (1 if direction == GPIO.HIGH else -1)
            self.update_gui()

    def door_control_loop(self):
        while self.is_running:
            time.sleep(1)
    
    def capture_loop(self):
        while self.is_running:
            frame = self.camera.capture_array()
            results = self.model.predict(frame, classes=0, conf=0.65, verbose=False)
            
            if len(results[0].boxes) > 0:  # คนหรือวัตถุถูกตรวจพบ
                self.move_door(self.DOOR_WIDTH)  # เปิดประตู
                time.sleep(4)
                
                while self.measure_distance() < self.SAFETY_DISTANCE:  # เช็คระยะห่างจากสิ่งกีดขวาง
                    time.sleep(1)
                
                self.move_door(0)  # ปิดประตู
                
            time.sleep(1)

    def update_gui(self):
        self.progress['value'] = self.door_position
        self.root.update()

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
