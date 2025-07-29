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

#the event loop creation
#loop = asyncio.get_event_loop()
#change_scene_task = loop.create_task()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame")
        break

    results = model(frame)
    results.render()
    annotated_frame = results.ims[0]
    cv2.imshow("Yanshee Object Detection", annotated_frame)
   #say objects names outloud
    names = results.pandas().xyxy[0]
    kotsy = names['name'].tolist()
    #for i in kotsy:
    #print(i)
    #YanAPI.sync_do_tts(kotsy)

    if "person" in kotsy:
        vis_task_res = YanAPI.get_visual_task_result("face", "recognition")
        if vis_task_res["data"]["recognition"]["name"] == "none":
            YanAPI.set_robot_led("camera", "red", "on")
            YanAPI.set_robot_led("button", "red", "on")
        else: 
            YanAPI.set_robot_led("camera", "green", "on")
            YanAPI.set_robot_led("button", "green", "on")
            YanAPI.start_play_motion("bow")
    else:
            YanAPI.set_robot_led("camera", "blue", "on")
            YanAPI.set_robot_led("button", "blue", "on")
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    #time.sleep(0.1)


cap.release()
cv2.destroyAllWindows()
YanAPI.close_vision_stream()

