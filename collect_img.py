import os
import cv2
from pathlib import Path


# скрипт для создания датасета для жестового языка для русского алфавита. размер датасета на букву 100 фоток

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
outpath = Path(Path.cwd(),'data')

alphabet = ['A', 'B', 'V', 'G', 'D', 'E', 'ZH', 'Z', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'Y', 'F',
            'H', 'TC', 'CH', 'SHA', 'SH', 'STONE', 'II', 'SOFT', 'AE', 'U', 'YA']
dataset_size = 100
flag_to_exit = False
cap = cv2.VideoCapture(0)
for symbol in alphabet:
    if not os.path.exists(os.path.join(DATA_DIR, symbol)):
        os.makedirs(os.path.join(DATA_DIR, symbol))
    print('Collecting data for class {}'.format(symbol))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Start {} ? Press "Q" '.format(symbol), (100, 50), cv2.FONT_HERSHEY_COMPLEX, 1.3,
                    (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
        if cv2.waitKey(25) == 27:
            flag_to_exit = True
            break
    if flag_to_exit:
        break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        path_to_save = Path(outpath, symbol, '{}.jpg'.format(str(counter)))
        cv2.imwrite(str(path_to_save), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
