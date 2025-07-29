import torch
import cv2
from ultralytics import YOLO
import YanAPI 
yanip = 12
print("starting")
YanAPI.yan_api_init(f"192.168.1.{yanip}")
print("intialized ip")
response = YanAPI.open_vision_stream()
print("Camera stream started...")

stream_url = f"http://192.168.1.{yanip}:8000/stream.mjpg" 
cap = cv2.VideoCapture(stream_url)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame")
        break

    results = model(frame)
    #say objects names outloud
    saw_person = False
    names = results.pandas().xyxy[0]
    kotsy = names['name'].tolist()
    for i in kotsy:
        print(i)
        if "person" in kotsy:
            saw_person = True 
        YanAPI.sync_do_tts(results.pandas().xyxy[i].name)

    if saw_person: 
        vis_task_res = YanAPI.get_visual_task_result("face", "recognition")
        if vis_task_res.data.recognition.name == "":
            YanAPI.set_robot_led("camera", "red", "on")
            YanAPI.set_robot_led("button", "red", "on")
        else: 
            YanAPI.set_robot_led("camera", "green", "on")
            YanAPI.set_robot_led("button", "green", "on")
    else:
            YanAPI.set_robot_led("camera", "blue", "on")
            YanAPI.set_robot_led("button", "blue", "on")
    results.render()
    annotated_frame = results.ims[0]

    cv2.imshow("Yanshee Object Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
YanAPI.close_vision_stream()

