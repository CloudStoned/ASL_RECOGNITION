import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.4)
DATA_DIR = 'data'

data = []
labels = []

def show_data(data):
    data_dict = pickle.load(open('data.pickle', 'rb'))
    print(data_dict['labels'])
    print(data_dict['data'])


def process_images():
    total_processed = 0
    class_count = 0
    class_stats = {}

    for dir_ in os.listdir(DATA_DIR):
        class_count += 1
        class_dir = os.path.join(DATA_DIR, dir_)
        file_count = len(os.listdir(class_dir))
        print(f"Processing class: {dir_} - Files in directory: {file_count}")
        
        class_processed = 0
        for img_path in os.listdir(class_dir):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(class_dir, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)
                class_processed += 1
                total_processed += 1

                if total_processed % 100 == 0:
                    print(f"Processed {total_processed} images")

            else:
                print(f"No hand detected in image: {img_path}")

        class_stats[dir_] = {"total": file_count, "processed": class_processed}
        print(f"Class {dir_}: Processed {class_processed} out of {file_count} images")
        print("-----------------------------------------------------")

    print(f"Total images processed: {total_processed}")
    print(f"Count of classes: {class_count}")
    print("\nClass-wise statistics:")
    for class_name, stats in class_stats.items():
        print(f"{class_name}: Processed {stats['processed']} out of {stats['total']} images")

    return data, labels

def save_data(data, labels):
    f = open('data.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Dataset created and saved as 'data.pickle'")
    f.close()

if __name__=='__main__':
#    data, labels = process_images()
#    print(f"\nLen of Data: {len(data)}")
#    print(f"Len of Labels: {len(labels)}")
#    print("\nSAVING DATA")
#    save_data(data, labels)
   show_data(data)