import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Define landmarks for eyes and iris detection
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]  # Left eye
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]  # Right eye
LEFT_IRIS_LANDMARKS = [469, 470, 471, 472]  # Left iris landmarks
RIGHT_IRIS_LANDMARKS = [474, 475, 476, 477]  # Right iris landmarks

# EAR threshold for detecting if eyes are open
eye_open_threshold = 0.15
closed_eye_count = 0
frames_threshold = 10

# Store last 10 frames' gaze status
gaze_history = []

# Tracking variables
session_active = False
session_start_time = None

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_landmarks):
    # Vertical distances
    vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Horizontal distance
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Function to calculate iris position relative to the eye
def calculate_iris_position(eye_landmarks, iris_landmarks):
    eye_width = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # Eye width
    iris_center = np.mean(iris_landmarks, axis=0)  # Iris center
    iris_to_left_corner = np.linalg.norm(iris_center - eye_landmarks[0])  # Distance from iris to left eye corner
    iris_position_ratio = iris_to_left_corner / eye_width  # Normalized position
    return iris_position_ratio

# Function to calculate head orientation (yaw, pitch, roll)
def calculate_orientation(landmarks):
    # Key landmarks
    nose_tip = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
    chin = np.array([landmarks[152].x, landmarks[152].y, landmarks[152].z])
    left_eye = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
    right_eye = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])
    
    # Yaw (horizontal turn: left/right)
    eye_vector = right_eye - left_eye
    yaw = np.arctan2(eye_vector[2], eye_vector[0]) * 180 / np.pi

    # Pitch (vertical tilt: up/down)
    face_vector = chin - nose_tip
    pitch = np.arctan2(face_vector[2], face_vector[1]) * 180 / np.pi

    # Roll (head tilt: clockwise/anticlockwise)
    roll = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi

    return yaw, pitch, roll

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get eye landmarks
            left_eye = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in LEFT_EYE_LANDMARKS])
            right_eye = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in RIGHT_EYE_LANDMARKS])

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            # Check if eyes are open
            if left_ear < eye_open_threshold and right_ear < eye_open_threshold:
                closed_eye_count += 1
                if closed_eye_count >= frames_threshold:
                    status = "Eyes Closed"
                    cv2.putText(frame, f"Status: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    continue  # Skip gaze detection if eyes are closed
            else:
                status = "Eyes Open"
                closed_eye_count = 0

            # Get iris landmarks
            left_iris = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in LEFT_IRIS_LANDMARKS])
            right_iris = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in RIGHT_IRIS_LANDMARKS])

            # Calculate iris positions
            left_iris_position = calculate_iris_position(left_eye, left_iris)
            right_iris_position = calculate_iris_position(right_eye, right_iris)

            # Calculate head orientation
            yaw, pitch, roll = calculate_orientation(face_landmarks.landmark)

            # Adjust gaze thresholds dynamically based on yaw
            left_min, left_max = 0.35, 0.65
            right_min, right_max = 0.35, 0.60
            min_offset, max_offset = 0, 0

            if yaw < -25:  # Looking left
                max_offset = 0.10
                min_offset = 0.05
            elif yaw > 20:  # Looking right
                max_offset = -0.10
                min_offset = -0.05

            # Determine gaze status for the current frame
            if (left_min + min_offset) < left_iris_position < (left_max + max_offset) and (right_min + min_offset) < right_iris_position < (right_max + max_offset):
                current_gaze_status = "Looking at Camera"
            else:
                current_gaze_status = "Looking Away"

            # Update gaze history
            gaze_history.append(current_gaze_status)
            if len(gaze_history) > 10:  # Keep only the last 10 frames
                gaze_history.pop(0)

            # Determine final gaze status based on the last 10 frames
            if gaze_history.count("Looking Away") >= 4:
                final_gaze_status = "Looking Away"
                if session_active:
                    session_active = False
                    session_end_time = time.time()
                    print(f"Session ended: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_end_time))}")
                    print(f"Session duration: {session_end_time - session_start_time:.2f} seconds")
            else:
                final_gaze_status = "Looking at Camera"
                if not session_active:
                    session_active = True
                    session_start_time = time.time()
                    print(f"Session started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_start_time))}")

            # Display gaze status and EAR values
            cv2.putText(frame, f"Status: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Gaze Status: {final_gaze_status}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw eye and iris landmarks
            for x, y in left_eye:
                cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
            for x, y in right_eye:
                cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
            for x, y in left_iris:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
            for x, y in right_iris:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Show the frame
    cv2.imshow("Iris and Gaze Detection with Eyes Open Check", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
