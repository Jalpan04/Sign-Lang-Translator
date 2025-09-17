import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os

# --- 1. SETUP AND INITIALIZATION ---

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Load the trained models for Left and Right hands
try:
    model_L = tf.keras.models.load_model('L_keypoint_classifier_final.h5')
    model_R = tf.keras.models.load_model('R_keypoint_classifier_final.h5')
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure 'L_keypoint_classifier_final.h5' and 'R_keypoint_classifier_final.h5' are in the same directory.")
    exit()

# Define the class labels in the correct alphabetical order
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']

# The sequence of letters for the tutorial
TUTORIAL_SEQUENCE = [chr(ord('A') + i) for i in range(26)] # A, B, C, ..., Z
WINDOW_NAME = 'ASL Alphabet Tutorial (MediaPipe)'

# --- 2. HELPER FUNCTION ---

def preprocess_landmarks(landmark_list):
    """
    Preprocesses landmarks to be relative and normalized.
    This is crucial for making the model robust to hand position and size.
    """
    landmarks = np.array([[lm.x, lm.y] for lm in landmark_list.landmark])
    base_x, base_y = landmarks[0]
    relative_landmarks = landmarks - [base_x, base_y]
    flat_landmarks = relative_landmarks.flatten()
    max_val = np.max(np.abs(flat_landmarks))
    if max_val > 0:
        normalized_landmarks = flat_landmarks / max_val
    else:
        normalized_landmarks = flat_landmarks
    return normalized_landmarks

# --- 3. MAIN APPLICATION LOGIC ---

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Tutorial state variables
    current_letter_index = 0
    correct_feedback_start_time = 0
    show_correct_feedback = False
    tutorial_complete = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Flip the frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        predicted_letter = ""
        confidence = 0.0

        # Check if a hand was detected
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label

            # Preprocess landmarks for the model
            processed_landmarks = preprocess_landmarks(hand_landmarks)
            model_input = np.expand_dims(processed_landmarks, axis=0)

            # Select the correct model and predict
            if handedness == 'Left':
                prediction = model_R.predict(model_input, verbose=0)
            else: # Right
                prediction = model_L.predict(model_input, verbose=0)

            predicted_class_index = np.argmax(prediction)
            predicted_letter = CLASS_NAMES[predicted_class_index]
            confidence = np.max(prediction) * 100

            # Draw the hand skeleton
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- UI and Tutorial Logic ---
        if not tutorial_complete:
            target_letter = TUTORIAL_SEQUENCE[current_letter_index]

            # Display the large target letter
            cv2.putText(frame, target_letter, (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 10)
            cv2.putText(frame, "Show me this sign", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Check if the user's sign is correct
            if predicted_letter == target_letter and confidence > 90: # Using a high confidence threshold
                if not show_correct_feedback:
                    show_correct_feedback = True
                    correct_feedback_start_time = time.time()

                cv2.putText(frame, "Correct!", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                # Move to the next letter after 2 seconds of correct feedback
                if time.time() - correct_feedback_start_time > 2:
                    show_correct_feedback = False
                    current_letter_index += 1
                    if current_letter_index >= len(TUTORIAL_SEQUENCE):
                        tutorial_complete = True
            else:
                show_correct_feedback = False
                # Display the live prediction if a hand is visible
                if predicted_letter:
                    text = f"Prediction: {predicted_letter} ({confidence:.1f}%)"
                    cv2.putText(frame, text, (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        else:
            # Tutorial complete message
            cv2.putText(frame, "Congratulations!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            cv2.putText(frame, "You completed the alphabet!", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display quit instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the final frame
        cv2.imshow(WINDOW_NAME, frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- 4. CLEANUP ---
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()