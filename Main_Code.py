import os
import random
import cv2
import numpy as np
from ultralytics import YOLO

# Create a directory to save detected objects 
save_dir = "F:\\Frames"  # Updated path for saving images
os.makedirs(save_dir, exist_ok=True)

# Open the class list file
class_file_path = "C:\\Users\\DELL\\Downloads\\Internship\\utils\\coco.txt"  # Updated path for class list file
with open(class_file_path, "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load a pretrained YOLOv8n model
model_weights_path = "C:\\Users\\DELL\\Downloads\\Internship\\weights\\yolov8n.pt"  # Updated path for YOLO weights
model = YOLO(model_weights_path, "v8")

# Video capture
video_path = "C:\\Users\\DELL\\Downloads\\Internship\\background video _ people _ walking _.mp4"  # Updated path for video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video file")
    exit()

frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]  # Returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # Save detected objects as images
            obj = frame[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
            cv2.imwrite(os.path.join(save_dir, f"frame_{frame_count}_obj_{i}.png"), obj)

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

    frame_count += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
