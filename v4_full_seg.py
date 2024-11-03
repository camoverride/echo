import cv2
import numpy as np
import mediapipe as mp
import pygame
import random



# Initialize MediaPipe Selfie Segmentation and Hands
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_hands = mp.solutions.hands
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame for audio
pygame.mixer.init()

# Load or generate audio files for each square (6x6 grid)
audio_files = [[f"audio/sound_{i}_{j}.wav" for j in range(6)] for i in range(6)]
sounds = {(i, j): pygame.mixer.Sound(audio_files[i][j]) for i in range(6) for j in range(6)}

# Set up video capture
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape

# Calculate the size of each grid cell based on screen dimensions
grid_rows, grid_cols = 6, 6
cell_height, cell_width = h // grid_rows, w // grid_cols

# Create a 6x6 grid of random colors that fills the screen
grid = np.zeros((h, w, 3), dtype=np.uint8)
colors = [[(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(grid_cols)] for _ in range(grid_rows)]
for i in range(grid_rows):
    for j in range(grid_cols):
        color = colors[i][j]
        cv2.rectangle(grid, (j * cell_width, i * cell_height), ((j + 1) * cell_width, (i + 1) * cell_height), color, -1)

# Function to detect which cell the index finger tip is over
def get_square(x, y):
    return y // cell_height, x // cell_width

# Main loop
current_square = None
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply Selfie Segmentation
    selfie_results = selfie_segmentation.process(frame_rgb)
    mask = (selfie_results.segmentation_mask > 0.5).astype(np.uint8) * 255

    # Extract the segmented person using the mask
    person_segment = cv2.bitwise_and(frame, frame, mask=mask)

    # Overlay the segmented person onto the grid
    display_frame = cv2.addWeighted(grid, 1, person_segment, 1, 0)

    # Process the frame with MediaPipe Hands for landmark detection
    hand_results = hands.process(frame_rgb)

    # If a hand is detected, draw landmarks and play sound based on fingertip location
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract index finger tip position for square detection
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            tip_x, tip_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Determine the square the index finger tip is over
            square = get_square(tip_x, tip_y)

            # If the index finger moves to a new square, play the audio file for that square
            if square != current_square:
                current_square = square
                if 0 <= square[0] < grid_rows and 0 <= square[1] < grid_cols:
                    sounds[square].play()

            # Draw a circle at the index finger tip location to represent the pointer
            cv2.circle(display_frame, (tip_x, tip_y), 10, (255, 255, 255), -1)

    # Display the output frame
    cv2.imshow('Selfie Segmentation with Hand Landmarks on Colored Grid', display_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
