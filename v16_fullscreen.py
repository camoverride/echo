import cv2
import numpy as np
import mediapipe as mp
import pygame
import random
import os
import subprocess

# Set environment variables for display and Pygame
os.environ["DISPLAY"] = ":0"  # Use the correct display
os.environ["SDL_VIDEODRIVER"] = "x11"

# Function to get monitor dimensions using xrandr
def get_monitor_dimensions():
    try:
        output = subprocess.check_output(['xrandr']).decode('utf-8')
        for line in output.splitlines():
            if '*' in line:  # Find the line with the current resolution
                resolution = line.split()[0]
                width, height = map(int, resolution.split('x'))
                return width, height
    except Exception as e:
        print(f"Error getting monitor dimensions: {e}")
    return None, None

# Get the monitor dimensions
display_width, display_height = get_monitor_dimensions()
if display_width is None or display_height is None:
    print("Could not retrieve monitor dimensions.")
    exit()

# Initialize MediaPipe components
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_hands = mp.solutions.hands
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame for audio
pygame.mixer.init()

# Load audio files
audio_folder = "audio"
audio_files = [f"{audio_folder}/{file}" for file in os.listdir(audio_folder) if file.endswith(".wav")]
random.shuffle(audio_files)
if len(audio_files) < 400:
    raise ValueError("There are not enough audio files. Please ensure there are exactly 100 audio files in the 'audio' folder.")
elif len(audio_files) > 400:
    audio_files = audio_files[:400]
sounds = {(i, j): pygame.mixer.Sound(audio_files.pop()) for i in range(20) for j in range(20)}

# Set the RTSP URL
rtsp_url = "rtsp://admin:admin123@192.168.0.217:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)  # Use FFMPEG backend to avoid display requirements

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

# Create a window named 'Selfie Segmentation' and set it to fullscreen
cv2.namedWindow('Selfie Segmentation with Hand Landmarks on Colored Grid', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Selfie Segmentation with Hand Landmarks on Colored Grid', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Main loop
current_square = None
while cap.isOpened():
    # Clear the buffer by grabbing frames to reach the latest frame
    for _ in range(5):  # Adjust the number of grabs if needed
        cap.grab()

    # Step 1: Read frame
    success, frame = cap.read()
    if not success:
        break

    # Mirror the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Resize frame to smaller dimensions for faster processing
    frame_small = cv2.resize(frame, (640, 360))  # Keep this smaller for processing
    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    # Apply Selfie Segmentation
    selfie_results = selfie_segmentation.process(frame_rgb)
    mask = (selfie_results.segmentation_mask > 0.5).astype(np.uint8) * 255
    person_segment = cv2.bitwise_and(frame_small, frame_small, mask=mask)

    # Create grid with the same size as the display frame
    grid = np.zeros((display_height, display_width, 3), dtype=np.uint8)  # Match the display size
    colors = [[(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(grid.shape[1])] for _ in range(grid.shape[0] // (display_height // 20))]
    for i in range(grid.shape[0] // (display_height // 20)):
        for j in range(grid.shape[1] // (display_width // 20)):
            color = colors[i][j]
            cv2.rectangle(grid, (j * (display_width // grid.shape[1]), i * (display_height // grid.shape[0])),
                          ((j + 1) * (display_width // grid.shape[1]), (i + 1) * (display_height // grid.shape[0])), color, -1)

    # Overlay segmented person onto the grid
    display_frame = cv2.addWeighted(grid, 1, person_segment, 1, 0)

    # Process the frame with MediaPipe Hands for landmark detection
    hand_results = hands.process(frame_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            tip_x, tip_y = int(index_finger_tip.x * 640), int(index_finger_tip.y * 360)
            square = get_square(tip_x, tip_y)
            if square != current_square:
                current_square = square
                if 0 <= square[0] < 20 and 0 <= square[1] < 20:
                    sounds[square].play()
            cv2.circle(display_frame, (tip_x, tip_y), 10, (255, 255, 255), -1)

    # Display the output frame in fullscreen
    cv2.imshow('Selfie Segmentation with Hand Landmarks on Colored Grid', display_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
