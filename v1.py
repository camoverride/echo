import cv2
import numpy as np
import mediapipe as mp
import pygame
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame for audio
pygame.mixer.init()

# Load or generate audio files for each square (6x6 grid)
audio_files = [[f"audio/sound_{i}_{j}.wav" for j in range(6)] for i in range(6)]

# Prepare audio for each square
sounds = {}
for i in range(6):
    for j in range(6):
        sounds[(i, j)] = pygame.mixer.Sound(audio_files[i][j])

# Create a 6x6 grid of random colors
grid_size = 6
square_size = 100
grid = np.zeros((grid_size * square_size, grid_size * square_size, 3), dtype=np.uint8)
colors = [[(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(6)] for _ in range(6)]
for i in range(grid_size):
    for j in range(grid_size):
        color = colors[i][j]
        cv2.rectangle(grid, (j * square_size, i * square_size), ((j + 1) * square_size, (i + 1) * square_size), color, -1)

# Function to detect which square the hand is over
def get_square(x, y):
    return y // square_size, x // square_size

# Main loop
cap = cv2.VideoCapture(0)
current_square = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Overlay the grid on the frame
    display_frame = frame.copy()
    display_frame[: grid.shape[0], : grid.shape[1]] = grid

    # If a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the wrist (or another central landmark) for basic hand position
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, _ = frame.shape
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

            # Determine the square the wrist is over
            square = get_square(wrist_x, wrist_y)

            # If the hand moves to a new square, play the audio file for that square
            if square != current_square:
                current_square = square
                if 0 <= square[0] < grid_size and 0 <= square[1] < grid_size:
                    sounds[square].play()

            # Draw a circle at the wrist location to represent hand position
            cv2.circle(display_frame, (wrist_x, wrist_y), 10, (255, 255, 255), -1)

    # Display the output frame
    cv2.imshow('Hand Tracking on Colored Grid', display_frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
