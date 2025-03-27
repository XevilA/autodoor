import RPi.GPIO as GPIO
import time
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# ตั้งค่า GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

# ตั้งค่าพินมอเตอร์
STEP_PIN = 13    # พินสเต็ป
DIR_PIN = 11     # พินทิศทาง
GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)

# ตั้งค่าเซนเซอร์วัดระยะ
TRIG_PIN = 16    # พินส่งสัญญาณ
ECHO_PIN = 18    # พินรับสัญญาณ
GPIO.setup(TRIG_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ECHO_PIN, GPIO.IN)

# ตั้งค่ากล้อง
camera = Picamera2()
config = camera.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
camera.configure(config)
camera.start()
time.sleep(2)  # รอให้เซ็นเซอร์พร้อมทำงาน

# โมเดล AI
model = YOLO('yolov8n.pt')

# ตั้งค่าพารามิเตอร์ระบบ
STEP_ANGLE = 1.8     # องศาต่อสเต็ป (เปลี่ยนตามสเปคมอเตอร์)
GEAR_RATIO = 10      # อัตราทดเกียร์
DESIRED_ANGLE = 90   # องศาที่ต้องการให้ประตูเปิด
STEP_DELAY = 0.003    # ความเร็วสเต็ป (วินาที)

# คำนวณจำนวนสเต็ป
TOTAL_STEPS = int((DESIRED_ANGLE * GEAR_RATIO) / STEP_ANGLE)  # สูตรคำนวณ

door_state = {
    "is_open": False,
    "in_motion": False,
    "safety_stop": False
}

def calculate_distance():
    try:
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)

        timeout = time.time() + 0.04
        start = end = time.time()

        while GPIO.input(ECHO_PIN) == 0 and time.time() < timeout:
            start = time.time()
        
        while GPIO.input(ECHO_PIN) == 1 and time.time() < timeout:
            end = time.time()

        return (end - start) * 17150  # คำนวณเป็นเซนติเมตร
    except:
        return 1000  # คืนค่าสูงสุดหากมีข้อผิดพลาด

def move_door(direction, steps):
    GPIO.output(DIR_PIN, direction)
    for _ in range(int(steps)):
        if door_state["safety_stop"]:
            break
        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(STEP_DELAY)
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(STEP_DELAY)

try:
    while True:
        # จับภาพและประมวลผล
        frame = camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, -1)  # พลิกภาพแนวตั้งและแนวนอน
        
        # ตรวจจับคน
        results = model.predict(frame, 
                              imgsz=320, 
                              classes=0,  # เฉพาะคลาสคน
                              verbose=False, 
                              conf=0.6)   # ความเชื่อมั่นขั้นต่ำ 60%
        
        people_detected = len(results[0].boxes) > 0

        # ตรวจสอบความปลอดภัย
        current_distance = calculate_distance()
        door_state["safety_stop"] = current_distance < 50  # หยุดถ้ามีสิ่งกีดขวางในระยะ 50 ซม.

        # ควบคุมประตู
        if not door_state["in_motion"]:
            if people_detected and not door_state["is_open"]:
                print("กำลังเปิดประตู...")
                door_state["in_motion"] = True
                move_door(GPIO.HIGH, TOTAL_STEPS)
                door_state["is_open"] = True
                door_state["in_motion"] = False
                
            elif not people_detected and door_state["is_open"]:
                if current_distance > 80:  # ตรวจสอบระยะปลอดภัย
                    print("กำลังปิดประตู...")
                    door_state["in_motion"] = True
                    move_door(GPIO.LOW, TOTAL_STEPS)
                    door_state["is_open"] = False
                    door_state["in_motion"] = False

        # แสดงข้อมูล
        status_text = f"สถานะ: {'เปิด' if door_state['is_open'] else 'ปิด'} | ตรวจจับคน: {len(results[0].boxes)}"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("ระบบประตูอัตโนมัติ", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"เกิดข้อผิดพลาด: {str(e)}")
finally:
    GPIO.output(DIR_PIN, GPIO.LOW)
    GPIO.output(STEP_PIN, GPIO.LOW)
    GPIO.cleanup()
    camera.stop()
    cv2.destroyAllWindows()
    print("ปิดระบบเรียบร้อย")
