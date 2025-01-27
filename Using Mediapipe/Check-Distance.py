import cv2
import mediapipe as mp
import time

class DistanceEstimator:
    def __init__(self, real_face_height=20):
        """
        Initializes the DistanceEstimator.
        :param real_face_height: Average height of a human face in cm (default: 20).
        """
        self.real_face_height = real_face_height
        self.session_active = False
        self.session_start_time = None

    def visualize(self, frame, results):
        """
        Visualizes the distance estimation results on the frame.
        :param frame: Input frame (BGR image).
        :param results: List of tuples containing distance, distance category, and key coordinates.
        """
        for distance, category, forehead_coords, chin_coords in results:
            cv2.circle(frame, forehead_coords, 5, (0, 255, 0), -1)
            cv2.circle(frame, chin_coords, 5, (0, 0, 255), -1)
            cv2.putText(frame, f'Distance: {int(distance)}',
                        (forehead_coords[0] + 10, forehead_coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f'Category: {category}',
                        (forehead_coords[0] + 10, forehead_coords[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    @staticmethod
    def get_distance_category(distance):
        """
        Categorizes the distance into predefined ranges.
        :param distance: Calculated distance.
        :return: String category of distance.
        """
        if distance < 20:
            return "Immediate"
        elif 20 <= distance < 45:
            return "Near"
        elif 45 <= distance < 90:
            return "Mid-range"
        elif 90 <= distance < 180:
            return "Far"
        else:
            return "Too Far"

    def compute_distance(self, face_landmarks, frame_shape):
        """
        Computes the distance using pre-computed face landmarks.
        :param face_landmarks: Processed face landmarks from Mediapipe FaceMesh.
        :param frame_shape: Tuple of (height, width) of the frame.
        :return: List of distance results.
        """
        if not face_landmarks:
            if self.session_active:
                self.end_session()
            return []

        h, w = frame_shape[:2]
        forehead = face_landmarks.landmark[10]
        chin = face_landmarks.landmark[152]

        forehead_coords = (int(forehead.x * w), int(forehead.y * h))
        chin_coords = (int(chin.x * w), int(chin.y * h))

        face_height_pixels = abs(forehead_coords[1] - chin_coords[1])
        if face_height_pixels > 0:  # Avoid division by zero
            distance = (w * self.real_face_height) / face_height_pixels
            distance_category = self.get_distance_category(distance)

            if distance_category not in ["Far", "Too Far"]:
                self.start_session()
            else:
                self.end_session()

            return [(distance, distance_category, forehead_coords, chin_coords)]
        
        if self.session_active:
            self.end_session()
        return []

    def start_session(self):
        """Starts the session if not already active."""
        if not self.session_active:
            self.session_start_time = time.time()
            self.session_active = True
            print(f"Session started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.session_start_time))}")

    def end_session(self):
        """Ends the session if currently active."""
        if self.session_active:
            session_end_time = time.time()
            duration = session_end_time - self.session_start_time
            print(f"Session ended: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_end_time))}")
            print(f"Session duration: {duration:.2f} seconds")
            self.session_active = False


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    estimator = DistanceEstimator()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)
        # Convert the frame to RGB for Mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Extract face landmarks if available
        face_landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None

        results = estimator.compute_distance(face_landmarks, frame.shape)
        estimator.visualize(frame, results)

        cv2.imshow('FaceMesh - Distance Estimation', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
