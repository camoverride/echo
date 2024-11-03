import cv2
import numpy as np
import mediapipe as mp
import pygame
import random
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
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

# Function to calculate distance between two landmarks
def calculate_distance(point1, point2):
    return int(math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))

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

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Create a copy of the grid to overlay the segmented hand and display landmarks
    display_frame = grid.copy()

    # If a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert landmarks to pixel coordinates
            landmark_points = [
                (int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark
            ]

            # Calculate relative thickness based on the hand size
            wrist = landmark_points[mp_hands.HandLandmark.WRIST.value]
            index_mcp = landmark_points[mp_hands.HandLandmark.INDEX_FINGER_MCP.value]
            relative_thickness = calculate_distance(wrist, index_mcp) // 4  # Adjust divisor to control thickness scaling
            dilation_size = relative_thickness // 2  # Adjust for smoother edges

            # Initialize mask
            mask = np.zeros((h, w), dtype=np.uint8)

            # Define finger segments (tubes) as connections between consecutive landmarks
            finger_tubes = [
                [mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_TIP],
                [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP],
                [mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                [mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_TIP],
                [mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP]
            ]

            # Draw tubes for each finger with dynamic thickness
            for finger in finger_tubes:
                for i in range(len(finger) - 1):
                    pt1 = landmark_points[finger[i].value]
                    pt2 = landmark_points[finger[i + 1].value]
                    cv2.line(mask, pt1, pt2, 255, thickness=relative_thickness)

            # Connect wrist and lower hand area to complete the hand shape
            wrist_points = [
                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            ]
            wrist_coords = np.array([(int(pt.x * w), int(pt.y * h)) for pt in wrist_points], np.int32)
            cv2.fillConvexPoly(mask, wrist_coords, 255)

            # Dilate mask further based on relative hand size
            mask = cv2.dilate(mask, np.ones((dilation_size, dilation_size), np.uint8), iterations=1)

            # Extract the hand region from the frame using the mask
            hand_segment = cv2.bitwise_and(frame, frame, mask=mask)

            # Overlay the segmented hand onto the grid
            display_frame = cv2.addWeighted(display_frame, 1, hand_segment, 1, 0)

            # Draw landmarks and connections on the hand segment
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
    cv2.imshow('Hand Segmentation and Audio Playback on Colored Grid', display_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
