import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

# Load the hand detection model
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# Define hand bone connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17)  # palm
]

while True:
    # Read a frame from webcam
    success, frame = cap.read()
    if success:
        # Convert BGR to RGB (MediaPipe needs RGB)
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB_frame)
        
        # Detect hands in the frame
        result = detector.detect(mp_image)
        
        # Draw landmarks if hand detected
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                h, w, _ = frame.shape
                
                # Draw green circles on each landmark point
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
                # Draw lines connecting the landmarks
                for connection in HAND_CONNECTIONS:
                    start = hand_landmarks[connection[0]]
                    end = hand_landmarks[connection[1]]
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Show the video feed
        cv2.imshow("capture image", frame)
        
        # Quit if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
