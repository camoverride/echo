import cv2
import numpy as np
import mediapipe as mp
import pygame
import random
import os

# Initialize MediaPipe components
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame for audio
pygame.mixer.init()

# Load audio files
audio_folder = "audio"
audio_files = [f"{audio_folder}/{file}" for file in os.listdir(audio_folder) if file.endswith(".wav")]
random.shuffle(audio_files)
if len(audio_files) < 100:
    raise ValueError("There are not enough audio files. Please ensure there are exactly 100 audio files in the 'audio' folder.")
elif len(audio_files) > 100:
    audio_files = audio_files[:100]
sounds = {(i, j): pygame.mixer.Sound(audio_files.pop()) for i in range(10) for j in range(10)}

# Set the RTSP URL
rtsp_url = "rtsp://admin:admin123@192.168.0.217:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

# Set processing frame size
processing_width = 640
processing_height = 360

# Retrieve original frame size for display
ret, frame = cap.read()
if not ret:
    print("Error: Unable to read from stream.")
    cap.release()
    exit()
display_height, display_width, _ = frame.shape

# Grid setup
grid_rows, grid_cols = 10, 10
cell_height, cell_width = processing_height // grid_rows, processing_width // grid_cols
grid = np.zeros((processing_height, processing_width, 3), dtype=np.uint8)
colors = [[(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(grid_cols)] for _ in range(grid_rows)]
for i in range(grid_rows):
    for j in range(grid_cols):
        color = colors[i][j]
        cv2.rectangle(grid, (j * cell_width, i * cell_height), ((j + 1) * cell_width, (i + 1) * cell_height), color, -1)

# Function to detect which cell the nose is over
def get_square(x, y):
    return y // cell_height, x // cell_width

# Main loop
current_square = None
while cap.isOpened():
    for _ in range(5):
        cap.grab()

    # Step 1: Read frame
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_small = cv2.resize(frame, (processing_width, processing_height))
    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    # Apply Selfie Segmentation
    selfie_results = selfie_segmentation.process(frame_rgb)
    mask = (selfie_results.segmentation_mask > 0.5).astype(np.uint8) * 255
    person_segment = cv2.bitwise_and(frame_small, frame_small, mask=mask)

    display_frame = cv2.addWeighted(grid, 1, person_segment, 1, 0)

    # Process the frame with MediaPipe Face Mesh for landmark detection
    face_results = face_mesh.process(frame_rgb)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            nose_landmark = face_landmarks.landmark[1]  # Index for the nose
            nose_x, nose_y = int(nose_landmark.x * processing_width), int(nose_landmark.y * processing_height)
            cv2.circle(display_frame, (nose_x, nose_y), 10, (255, 255, 255), -1)  # Draw nose point

            square = get_square(nose_x, nose_y)
            if square != current_square:
                current_square = square
                if 0 <= square[0] < grid_rows and 0 <= square[1] < grid_cols:
                    sounds[square].play()

    # Resize display_frame back to original size for viewing
    display_frame_large = cv2.resize(display_frame, (display_width, display_height))

    # Display the output frame
    cv2.imshow('Selfie Segmentation with Nose Tracking on Colored Grid', display_frame_large)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
