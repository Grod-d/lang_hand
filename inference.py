import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'А', 1: 'Б', 2: 'Д',3: 'Э', 4: 'Ч', 5: 'Е',6: 'Ф', 7: 'Г', 8: 'Х',9: 'И', 10: 'Й', 11: 'К',
               12: 'Л', 13: 'М',14: 'Н', 15: 'О', 16: 'П',17: 'Р', 18: 'С', 19: 'Ш',20: 'Ь', 21: 'Ъ', 22: 'Т',
               23: 'Ц', 24: 'Ю', 25: 'В',26: 'У', 27: 'Я', 28: 'З',29: 'Ж'}
while True:

    data_aux = []
    input_ = []
    output_ = []

    ret, frame = cap.read()

    higth, wight, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  
                hand_landmarks,  
                mp_hands.HAND_CONNECTIONS,  
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                input_.append(x)
                output_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(input_))
                data_aux.append(y - min(output_))

        x1 = int(min(input_) * wight) - 10
        y1 = int(min(output_) * higth) - 10

        x2 = int(max(input_) * wight) - 10
        y2 = int(max(output_) * higth) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()