import asyncio
import time
import torch
import cv2
from ultralytics import YOLO
import YanAPI
import warnings
warnings.filterwarnings('ignore')

yanip = 12
print("starting")
YanAPI.yan_api_init(f"192.168.1.{yanip}")
print("intialized ip")
response = YanAPI.open_vision_stream()
print("Camera stream started...")

stream_url = f"http://192.168.1.{yanip}:8000/stream.mjpg" 
cap = cv2.VideoCapture(stream_url)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
saw_person = False
scene_changed = True
prev_gyro = 0
while cap.isOpened(): 
    if YanAPI.get_current_motion_play_state()["data"]["status"] == "idle":
        gyro_data = YanAPI.get_sensors_gyro()
        curr_gyro = gyro_data["data"]["gyro"][0]["gyro-x"] + gyro_data["data"]["gyro"][0]["gyro-y"] + gyro_data["data"]["gyro"][0]["gyro-z"]
        print(prev_gyro)
        print(curr_gyro)
        diff = abs(prev_gyro - curr_gyro)
        if diff > 5 and diff < 10: 
            scene_changed = True
        saw_person_changed = False
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame")
            break

        results = model(frame)
        results.render()
        annotated_frame = results.ims[0]
        cv2.imshow("Yanshee Object Detection", annotated_frame)
        if scene_changed: 
            print("scene_changed")
       #say objects names outloud

        names = results.pandas().xyxy[0]
        kotsy = names['name'].tolist()
        if scene_changed:
            for i in kotsy:
                YanAPI.sync_do_tts(i)

        if "person" in kotsy:
            saw_person_changed = not saw_person
            saw_person = True 
        else: 
            saw_person_changed = saw_person
            saw_person = False
        if saw_person_changed:
            print("changed")
            if saw_person: 
                YanAPI.start_play_motion("bow")
                vis_task_res = YanAPI.get_visual_task_result("face", "recognition")
                if vis_task_res["data"]["recognition"]["name"] == "none":
                    YanAPI.set_robot_led("camera", "red", "on")
                    YanAPI.set_robot_led("button", "red", "on")
                else: 
                    YanAPI.set_robot_led("camera", "green", "on")
                    YanAPI.set_robot_led("button", "green", "on")
            else:
                    YanAPI.set_robot_led("camera", "blue", "on")
                    YanAPI.set_robot_led("button", "blue", "on")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        #time.sleep(0.1)
        scene_changed = False
        prev_gyro = curr_gyro


cap.release()
cv2.destroyAllWindows()
YanAPI.close_vision_stream()

