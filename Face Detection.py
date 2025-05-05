import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
print('test')
print('test')
print('test')

# 3D model points for head pose estimation
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye
    (225.0, 170.0, -135.0),      # Right eye
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Calculate Eye Aspect Ratio (for eye closure detection)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to exit")

photo_taken = False  # Flag to save image only once

# Reduce webcam resolution to improve performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Button parameters
button_top_left = (10, 10)
button_bottom_right = (150, 50)

def draw_button(frame):
    cv2.rectangle(frame, button_top_left, button_bottom_right, (255, 0, 0), -1)  # Draw button
    cv2.putText(frame, "Click to Capture", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def is_button_clicked(x, y):
    return button_top_left[0] < x < button_bottom_right[0] and button_top_left[1] < y < button_bottom_right[1]

# Mouse click callback function to check if the button is clicked
def click_event(event, x, y, flags, param):
    global photo_taken
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_button_clicked(x, y) and not photo_taken:
            # Save the photo when button is clicked
            filename = "captured_face.jpg"
            cv2.imwrite(filename, frame)
            cv2.putText(frame, "Photo Captured", (button_top_left[0], button_bottom_right[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            photo_taken = True

# Set up mouse callback
cv2.setMouseCallback("Head Pose with Emotion Detection (Mirror View)", click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb = small_frame[:, :, ::-1]
    h, w = frame.shape[:2]

    # Detect faces and landmarks
    face_locations = face_recognition.face_locations(rgb)
    landmarks_list = face_recognition.face_landmarks(rgb)

    for face_location, lm in zip(face_locations, landmarks_list):
        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)

        image_points = np.array([
            lm['nose_tip'][2],
            lm['chin'][8],
            lm['left_eye'][0],
            lm['right_eye'][3],
            lm['top_lip'][0],
            lm['top_lip'][6]
        ], dtype="double") * 4

        # Eye detection
        left_eye = np.array(lm['left_eye'], dtype="double")
        right_eye = np.array(lm['right_eye'], dtype="double")

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        EAR_THRESHOLD = 0.2
        if left_ear < EAR_THRESHOLD:
            cv2.putText(frame, "WARNING: Left Eye Closed!", (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if right_ear < EAR_THRESHOLD:
            cv2.putText(frame, "WARNING: Right Eye Closed!", (left, top - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Emotion Detection - Run every 5th frame to improve performance
        if cv2.getTickCount() % 5 == 0:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion_dict = result[0]['emotion']
                dominant_emotion = result[0]['dominant_emotion']

                cv2.putText(frame, f"Dominant: {dominant_emotion}", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                y_offset = bottom + 60
                for emotion, score in emotion_dict.items():
                    text = f"{emotion}: {score:.1f}%"
                    cv2.putText(frame, text, (left, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    y_offset += 20

            except Exception as e:
                print("Emotion detection failed:", e)

        # Head Pose Estimation
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            x_rot = np.arctan2(rmat[2, 1], rmat[2, 2])
            y_rot = np.arctan2(-rmat[2, 0], sy)
            z_rot = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            x_rot = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y_rot = np.arctan2(-rmat[2, 0], sy)
            z_rot = 0

        pitch = np.degrees(x_rot)
        yaw = np.degrees(y_rot)
        roll = np.degrees(z_rot)

        text = f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}"
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw the button on the frame
    draw_button(frame)

    # Display the resulting frame
    cv2.imshow("Head Pose with Emotion Detection (Mirror View)", frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
