import cv2, os
from pathlib import Path

outpath = Path.cwd() /'data'
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow('frame', frame)
cv2.waitKey(250)
path_to_save=Path(Path.cwd(),'data','A','{}.jpg'.format(str(0)))
cv2.imwrite('test1.jpg', frame)