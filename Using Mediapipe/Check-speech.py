import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict, deque
from sortTrack import Sort
import time

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True, min_detection_confidence=0.5)

# Define mouth landmarks (inner lips)
INNER_LIPS_TOP = 13  # Top center of inner lips
INNER_LIPS_BOTTOM = 14  # Bottom center of inner lips

# Function to calculate the vertical distance between inner lip points
def calculate_inner_lip_distance(top_lip, bottom_lip):
    return np.linalg.norm(top_lip - bottom_lip)

# Thresholds and history parameters
MOVEMENT_THRESHOLD = 1.2
SPEECH_THRESHOLD = 2.0
FRAME_HISTORY = 10
EMA_ALPHA = 0.3
RATE_OF_CHANGE_THRESHOLD = 0.5
SILENCE_THRESHOLD = 2  # Seconds of silence to mark end of speech
CONSECUTIVE_SPEECH_FRAMES = 10  # Minimum consecutive frames to confirm speaking

# Movement history per face
lip_distances_history = defaultdict(lambda: deque(maxlen=FRAME_HISTORY))
smoothed_lip_distances = defaultdict(lambda: None)

# Speech event tracking
speech_events = defaultdict(lambda: {"speaking": False, "start_time": None, "last_speaking_time": None, "speech_frames": 0})

# Initialize SORT Tracker
mot_tracker = Sort(max_age=30, min_hits=3)

# Open video capture (webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and preprocess the frame
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe FaceMesh
    results = face_mesh.process(rgb_frame)

    detections = []  # List to store bounding boxes for SORT

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            cx_min, cy_min = w, h
            cx_max, cy_max = 0, 0

            for lm in face_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cx_min = min(cx_min, cx)
                cy_min = min(cy_min, cy)
                cx_max = max(cx_max, cx)
                cy_max = max(cy_max, cy)

            confidence_score = 1.0
            detections.append([cx_min, cy_min, cx_max, cy_max, confidence_score])

        detections = np.array(detections)
        tracked_objects_lips = mot_tracker.update(detections)

        for tracker in tracked_objects_lips:
            x1, y1, x2, y2, obj_id_lips = tracker
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            for face_landmarks in results.multi_face_landmarks:
                top_lip = np.array([face_landmarks.landmark[INNER_LIPS_TOP].x * w, face_landmarks.landmark[INNER_LIPS_TOP].y * h])
                bottom_lip = np.array([face_landmarks.landmark[INNER_LIPS_BOTTOM].x * w, face_landmarks.landmark[INNER_LIPS_BOTTOM].y * h])

                lip_distance = calculate_inner_lip_distance(top_lip, bottom_lip)
                previous_lip_distance = smoothed_lip_distances[obj_id_lips]
                smoothed_lip_distance = lip_distance if previous_lip_distance is None else EMA_ALPHA * lip_distance + (1 - EMA_ALPHA) * previous_lip_distance

                smoothed_lip_distances[obj_id_lips] = smoothed_lip_distance
                lip_distances_history[obj_id_lips].append(smoothed_lip_distance)

                average_lip_distance = np.mean(lip_distances_history[obj_id_lips])
                lip_distances_variation = np.std(lip_distances_history[obj_id_lips])

                # Detect speech activity
                if average_lip_distance > SPEECH_THRESHOLD and lip_distances_variation > RATE_OF_CHANGE_THRESHOLD:
                    speech_events[obj_id_lips]["speech_frames"] += 1

                    # Confirm speaking only after consecutive frames
                    if speech_events[obj_id_lips]["speech_frames"] >= CONSECUTIVE_SPEECH_FRAMES:
                        if not speech_events[obj_id_lips]["speaking"]:
                            speech_events[obj_id_lips]["speaking"] = True
                            speech_events[obj_id_lips]["start_time"] = time.time()
                            print(f"Person {obj_id_lips} started speaking at {speech_events[obj_id_lips]['start_time']:.2f}s")

                        speech_events[obj_id_lips]["last_speaking_time"] = time.time()
                    speech_status = "Speaking"
                else:
                    speech_events[obj_id_lips]["speech_frames"] = 0  # Reset speech frame count
                    if speech_events[obj_id_lips]["speaking"]:
                        time_since_last_speech = time.time() - speech_events[obj_id_lips]["last_speaking_time"]
                        if time_since_last_speech > SILENCE_THRESHOLD:
                            speech_events[obj_id_lips]["speaking"] = False
                            end_time = time.time()
                            duration = end_time - speech_events[obj_id_lips]["start_time"]
                            print(f"Person {obj_id_lips} stopped speaking at {end_time:.2f}s, duration: {duration:.2f}s")

                    speech_status = "Silent"

                # Visualization: Draw lip points and speech status
                cv2.circle(frame, (int(top_lip[0]), int(top_lip[1])), 3, (0, 255, 0), -1)  # Top lip point
                cv2.circle(frame, (int(bottom_lip[0]), int(bottom_lip[1])), 3, (0, 0, 255), -1)  # Bottom lip point
                cv2.putText(frame, f"Speech Status: {speech_status}", (center_x, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Show the frame
    cv2.imshow("Multi-Face Speech Detection with SORT Tracking", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
