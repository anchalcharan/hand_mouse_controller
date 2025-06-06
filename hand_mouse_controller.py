import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Variables for smoothing
prev_x, prev_y = 0, 0
smooth_factor = 0.8

# Variables for click detection
click_threshold = 0.05
last_click_time = 0
click_cooldown = 0.5  # seconds
pause_after_click = 1.0  # seconds to pause after clicking

def map_coordinates(x, y, frame_width, frame_height):
    """Map hand coordinates to screen coordinates"""
    # Map to screen coordinates (removed the x flip)
    screen_x = int(x * screen_width)
    screen_y = int(y * screen_height)
    return screen_x, screen_y

def smooth_movement(current_x, current_y, prev_x, prev_y):
    """Apply smoothing to mouse movement"""
    # More direct movement with less smoothing
    smoothed_x = int(current_x)
    smoothed_y = int(current_y)
    return smoothed_x, smoothed_y

def is_click_gesture(thumb_tip, index_tip):
    """Detect if the gesture is a click (thumb and index finger close together)"""
    distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
    return distance < click_threshold

print("Hand Mouse Controller Started!")
print("Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # Only process the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

            # Get index finger tip coordinates
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Map coordinates to screen
            screen_x, screen_y = map_coordinates(
                index_tip.x, 
                index_tip.y,
                frame.shape[1],
                frame.shape[0]
            )

            # Apply smoothing
            screen_x, screen_y = smooth_movement(screen_x, screen_y, prev_x, prev_y)
            prev_x, prev_y = screen_x, screen_y

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Check for click gesture
            current_time = time.time()
            if (is_click_gesture(
                (thumb_tip.x, thumb_tip.y),
                (index_tip.x, index_tip.y)
            ) and current_time - last_click_time > click_cooldown):
                pyautogui.click()
                last_click_time = current_time
                # Pause briefly after clicking
                time.sleep(pause_after_click)

        # Display the frame
        cv2.imshow('Hand Mouse Controller', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    hands.close() 