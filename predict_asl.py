import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# --- SETUP ---
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # We will process one hand at a time
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Load the trained models for Left and Right hands
try:
    model_L = tf.keras.models.load_model('L_keypoint_classifier_final.h5')
    model_R = tf.keras.models.load_model('R_keypoint_classifier_final.h5')
except Exception as e:
    print(f"Error loading models: {e}")
    print(
        "Please ensure 'L_keypoint_classifier_final.h5' and 'R_keypoint_classifier_final.h5' are in the same directory.")
    exit()

# Create a dictionary to map the class indices to letters
# Make sure this matches the labels used during training
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']


# --- HELPER FUNCTION ---
def preprocess_landmarks(landmark_list):
    """
    Preprocesses landmarks to be relative and normalized.
    This function is crucial for making the model robust to hand position and size.
    """
    # Convert to a NumPy array
    landmarks = np.array([[lm.x, lm.y] for lm in landmark_list.landmark])

    # Make coordinates relative to the wrist (landmark 0)
    base_x, base_y = landmarks[0]
    relative_landmarks = landmarks - [base_x, base_y]

    # Flatten the array
    flat_landmarks = relative_landmarks.flatten()

    # Normalize the vector to have a max value of 1
    max_val = np.max(np.abs(flat_landmarks))
    if max_val > 0:
        normalized_landmarks = flat_landmarks / max_val
    else:
        normalized_landmarks = flat_landmarks

    return normalized_landmarks


# --- MAIN LOOP ---
# Start the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # For processing, convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    predicted_letter = ""

    # Check if a hand was detected
    if results.multi_hand_landmarks and results.multi_handedness:
        # Process the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label

        # Preprocess the landmarks
        processed_landmarks = preprocess_landmarks(hand_landmarks)
        model_input = np.expand_dims(processed_landmarks, axis=0)

        # Select the correct model based on handedness
        if handedness == 'Left':
            prediction = model_R.predict(model_input, verbose=0)  # Use Right model for Left hand in mirrored view
        else:  # Right
            prediction = model_L.predict(model_input, verbose=0)  # Use Left model for Right hand in mirrored view

        predicted_class_index = np.argmax(prediction)
        predicted_letter = class_names[predicted_class_index]

        # Draw the hand landmarks
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Draw the prediction on the frame
        # First, find the bounding box of the hand to place the text above it
        h, w, _ = frame.shape
        x_min = int(min([lm.x * w for lm in hand_landmarks.landmark]))
        y_min = int(min([lm.y * h for lm in hand_landmarks.landmark]))

        cv2.putText(frame, f"Prediction: {predicted_letter}", (x_min, y_min - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('ASL Keypoint Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
hands.close()
cap.release()
cv2.destroyAllWindows()
