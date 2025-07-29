import cv2
from ultralytics import YOLO

model = YOLO("yolov5x-cls.pt")

# Replace with Yanshee's stream URL
stream_url = "http://192.168.1.24:8080/video"
cap = cv2.VideoCapture(stream_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Stream not available.")
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("Yanshee IP YOLO", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


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