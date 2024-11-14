import cv2
import mediapipe as mp
import keyboard 

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# OpenCV capture
cap = cv2.VideoCapture(0)

# Cooldown state for play/pause gesture
play_pause_triggered = False

def detect_gesture(hand_landmarks):
    global play_pause_triggered

    fingers = []

    # Thumb
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x:
        fingers.append(1)
    else:
        fingers.append(0) 

    # Index Finger
    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y:
        fingers.append(1)  
    else:
        fingers.append(0) 

    # Middle Finger
    if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y:
        fingers.append(1)  # Middle finger is open
    else:
        fingers.append(0)  # Middle finger is closed

    # Ring Finger
    if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y:
        fingers.append(1)  # Ring finger is open
    else:
        fingers.append(0)  # Ring finger is closed

    # Pinky Finger
    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y:
        fingers.append(1)  # Pinky finger is open
    else:
        fingers.append(0)  # Pinky finger is closed

    # Check for play/pause gesture (index finger up, others down) with cooldown
    if fingers == [0, 1, 0, 0, 0] and not play_pause_triggered:
        keyboard.send("play/pause media")
        play_pause_triggered = True
    elif fingers != [0, 1, 0, 0, 0]:  # Reset play/pause trigger on neutral gesture
        play_pause_triggered = False

    # Volume up gesture (all fingers up)
    if fingers == [1, 1, 1, 1, 1]:
        keyboard.send("volume up")  # Volume Up

    # Volume down gesture (thumb and pinky up, others down)
    if fingers == [1, 0, 0, 0, 1]:
        keyboard.send("volume down")  # Volume Down

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    # Convert the frame color to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame with MediaPipe Hands
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gesture and control media
            detect_gesture(hand_landmarks)

    # Display the frame
    cv2.imshow("Hand Gesture Media Controller", frame)

    # Break with 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
