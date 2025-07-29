import cv2
from ultralytics import YOLO
import YanAPI 

YanAPI.yan_api_init("192.168.1.24")

response = YanAPI.open_vision_stream(resolution="640x480")
print("Camera stream started...")

stream_url = "http://192.168.1.24:8000/stream.mjpg" 
cap = cv2.VideoCapture(stream_url)


model = YOLO("yolov5x-cls.pt")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame")
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("Yanshee Object Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
YanAPI.close_vision_stream()



'''
If I wanna test on taken image ? 
use this command -> results = model("image.jpg")
annotated_frame = results[0].plot()
cv2.imshow("Yanshee Vision - YOLO Detection", annotated_frame) 

wanna test on a folder of images ? 
add '*' before 'jpg'

wanna test on taken vid ? 
cap = cv2.VideoCapture("vid.mp4")

'''