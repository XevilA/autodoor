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
        self.SAFETY_DISTANCE = 80        
        self.STEP_DELAY = 0.0005      

        self.mm_per_rev = self.BELT_PITCH * self.PULLEY_TEETH
        self.mm_per_step = self.mm_per_rev / self.STEPS_PER_REV
        self.total_steps = int(self.DOOR_WIDTH / self.mm_per_step)

        self.root = root
        self.setup_gui()

        self.door_position = 0.0       
        self.is_running = True
        self.manual_mode = False
        self.emergency_stop = False  # Flag for emergency stop
        
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

        # Emergency Stop Button
        ttk.Button(self.root, text="Emergency Stop", command=self.emergency_stop_function).pack()

    def measure_distance(self):
        readings = []
        for _ in range(3):
            GPIO.output(self.TRIG_PIN, True)
            time.sleep(0.00001)
            GPIO.output(self.TRIG_PIN, False)

            start = time.time()
            while GPIO.input(self.ECHO_PIN) == 0:
                start = time.time()

            end = time.time()
            while GPIO.input(self.ECHO_PIN) == 1:
                end = time.time()

            elapsed = end - start
            distance = (elapsed * 17150)  
            readings.append(distance)
            time.sleep(0.01)

        return sum(readings) / len(readings)

    def move_door(self, target_mm):
        if self.emergency_stop:  # Stop if emergency is triggered
            return
        
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
            time.sleep(0.1)

    def capture_loop(self):
        while self.is_running:
            if self.emergency_stop:  # Stop loop if emergency stop is triggered
                break

            frame = self.camera.capture_array()
            results = self.model.predict(frame, classes=0, conf=0.65, verbose=False)

            if len(results[0].boxes) > 0:
                self.move_door(self.DOOR_WIDTH)
                start_time = time.time()
                while time.time() - start_time < 4:  # Delay 4 seconds after opening
                    if self.emergency_stop:  # Stop if emergency stop is triggered
                        break

                    distance = self.measure_distance()
                    if distance > self.SAFETY_DISTANCE:
                        self.move_door(0)
                        break
                    time.sleep(0.2)

                if self.door_position > 0 and not self.emergency_stop:
                    self.move_door(0)

            time.sleep(0.5)

    def update_gui(self):
        self.progress['value'] = self.door_position
        self.root.update()

    def emergency_stop_function(self):
        self.emergency_stop = True
        GPIO.output(self.STEP_PIN, GPIO.LOW)
        self.door_position = 0
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
