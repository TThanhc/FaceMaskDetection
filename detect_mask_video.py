import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Load file model 
print("[INFO] Loading model detector...")
maskNet = load_model("mask_detector.h5")

# Initiate MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret: break

    # MediaPipe - RGB, OpenCV - BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Get size of frame
    (h, w) = frame.shape[:2]

    # if file a face in frame
    if results.detections:
        for detection in results.detections:
            # Get coordinates bounding box from MediaPipe (return scale 0-1)
            bboxC = detection.location_data.relative_bounding_box
            startX = int(bboxC.xmin * w)
            startY = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)
            endX = startX + width
            endY = startY + height

            # Ensure box in frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # cut the frame of face (ROI - Region of Interest)
            face = frame[startY:endY, startX:endX]
            
            # nothing
            if face.shape[0] < 1 or face.shape[1] < 1: continue

            # Preprocessing
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Predict
            (mask, withoutMask) = maskNet.predict(face)[0]

            # Labeling and define color
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255) # Xanh lá hoặc Đỏ

            # Percentage
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Draw frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Show
    cv2.imshow("Face Mask Detector", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()