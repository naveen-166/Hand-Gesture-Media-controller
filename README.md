
# Hand Gesture Media Controller

This project utilizes **OpenCV** and **MediaPipe** to create a hand gesture recognition system that can control media playback and volume on your computer.

## Features

- **Play/Pause Media**: Raise your index finger (others down).
- **Volume Up**: Raise all fingers.
- **Volume Down**: Raise your thumb and pinky (others down).
- **Real-time Video Feed**: Visualizes detected gestures using OpenCV.

## Requirements

Ensure you have the following dependencies installed:

- Python 3.7+
- OpenCV
- MediaPipe
- Keyboard

To install the required libraries, run:

```bash
pip install opencv-python mediapipe keyboard
```

## How It Works

1. **MediaPipe Hands**: Detects hand landmarks in real time.
2. **Gesture Detection**: Identifies specific hand gestures using landmark positions.
3. **Keyboard Control**: Maps gestures to keyboard shortcuts to control media.

### Gesture Mapping

| Gesture                    | Action                |
|----------------------------|-----------------------|
| Index finger up            | Play/Pause media      |
| All fingers up             | Volume up            |
| Thumb and pinky up         | Volume down          |

## How to Run

1. Clone this repository.
2. Run the script using the command:

   ```bash
   python hand_gesture_media_controller.py
   ```

3. Ensure your webcam is connected.
4. Use the gestures described above to control your media player.
5. Press the `q` key to exit the program.

## Preview

The program will display a live video feed from your webcam with detected hand landmarks drawn in real-time.

## Notes

- **Gesture Cooldown**: Prevents repeated triggering of the same action unless the gesture is reset.
- Ensure you have a media player running that responds to standard keyboard shortcuts like "play/pause media" or "volume up/down".
