import os
import cv2
import mediapipe as mp

DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 11
dataset_size = 300

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.4)

cap = cv2.VideoCapture(0)

def draw_landmarks(frame):
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hand landmarks
    results = hands.process(rgb_frame)
    
    # Create a copy of the frame to draw landmarks on
    frame_with_landmarks = frame.copy()
    
    # Draw the hand annotations on the copied image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_with_landmarks,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            )
    return frame_with_landmarks

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        preview_frame = draw_landmarks(frame)  # Draw landmarks for preview
        cv2.putText(preview_frame, 'Ready? Press "Q" to start collecting!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', preview_frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        preview_frame = draw_landmarks(frame)  # Draw landmarks for preview
        
        cv2.putText(preview_frame, 'Collecting frame {} of {}'.format(counter, dataset_size), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', preview_frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)  # Save original frame

        counter += 1

    print(f"Finished collecting data for class {j}")

cap.release()
cv2.destroyAllWindows()
print("Data collection complete!")