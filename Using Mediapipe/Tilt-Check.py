import cv2
import mediapipe as mp
import time

# Initialize Mediapipe FaceMesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path

# Tracking variables
session_active = False
session_start_time = None
session_end_time = None

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=10,  # Adjust for more faces if needed
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Mirror the image
        img = cv2.flip(img, 1)

        # Convert image to RGB for Mediapipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(imgRGB)

        # Get frame dimensions and center
        h, w, _ = img.shape
        frame_center_x, frame_center_y = w // 2, h // 2

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                # Draw facial landmarks

                # Initialize bounding box coordinates
                cx_min, cy_min = w, h  # Start with max possible values
                cx_max, cy_max = 0, 0  # Start with min possible values

                # Calculate bounding box by iterating over all landmarks
                for lm in faceLms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cx_min = min(cx_min, cx)
                    cy_min = min(cy_min, cy)
                    cx_max = max(cx_max, cx)
                    cy_max = max(cy_max, cy)

                # Calculate center of bounding box
                center_box_x = (cx_min + cx_max) // 2
                center_box_y = (cy_min + cy_max) // 2

                # Draw bounding box
                cv2.rectangle(img, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)

                # Calculate percentages relative to frame center
                offset_x = center_box_x - frame_center_x
                offset_y = center_box_y - frame_center_y
                percentage_x = (offset_x / (w/2)) * 100
                percentage_y = (offset_y / (h/2)) * 100

                # Determine position labels
                horizontal_position = "Right" if percentage_x > 0 else "Left"
                vertical_position = "Down" if percentage_y > 0 else "Up"

                # Display the position and percentage information
                cv2.putText(img, f"Horizontal: {horizontal_position} ({abs(percentage_x):.1f}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, f"Vertical: {vertical_position} ({abs(percentage_y):.1f}%)",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw the center of the bounding box
                cv2.circle(img, (center_box_x, center_box_y), 5, (0, 255, 0), -1)

        # Draw the center of the frame for reference
        cv2.circle(img, (frame_center_x, frame_center_y), 5, (0, 0, 255), -1)
        cv2.putText(img, "Frame Center (Origin)", 
                    (frame_center_x - 100, frame_center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Check if faces are detected
        if results.multi_face_landmarks:
            if not session_active:
                session_active = True
                session_start_time = time.time()
                print(f"Session started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_start_time))}")

        else:
            if session_active:
                session_active = False
                session_end_time = time.time()
                duration = session_end_time - session_start_time
                print(f"Session ended: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_end_time))}")
                print(f"Session duration: {duration:.2f} seconds")

        # Show the frame
        cv2.imshow('Face Mesh with Session Logging', img)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
