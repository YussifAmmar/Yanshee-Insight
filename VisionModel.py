import asyncio
from datetime import datetime
import base64
import time
import torch
import cv2
from ultralytics import YOLO
import YanAPI
import warnings
warnings.filterwarnings('ignore')

yanip = 26
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
frame_count = 0
while cap.isOpened(): 
    then = datetime.now()
    frame_count+=1
    if frame_count == 16 and YanAPI.get_current_motion_play_state()["data"]["status"] == "idle":
       # gyro_data = YanAPI.get_sensors_gyro()
       # curr_gyro = gyro_data["data"]["gyro"][0]["gyro-x"] + gyro_data["data"]["gyro"][0]["gyro-y"] + gyro_data["data"]["gyro"][0]["gyro-z"]
       # diff = abs(prev_gyro - curr_gyro)
       # if diff > 5 and diff < 10: 
       #     scene_changed = True
        frame_count = 0
        now = datetime.now()
        print((now - then).total_seconds())
        saw_person_changed = False
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame")
            break

        org_frame = frame.copy()
        results = model(frame)
        results.render()
        annotated_frame = results.ims[0]
        cv2.imshow("Yanshee Object Detection", annotated_frame)

        if scene_changed: 
            resized_frame = cv2.resize(org_frame, (200, round(200 / 1.3)))
            cv2.imwrite('image.jpeg',resized_frame)
            with open("image.jpeg", "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read())
            print(image_base64)
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
                #YanAPI.start_play_motion("bow")
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
        scene_changed = False
        #prev_gyro = curr_gyro


cap.release()
cv2.destroyAllWindows()
YanAPI.close_vision_stream()

