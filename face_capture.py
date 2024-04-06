import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

IMG_PATH = './data/test_images/'
# biến count để đếm số lượng ảnh
count = 50 
usr_name = input("Input ur name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)
# biến leap là bước nhảy, sau mỗi bước nhảy sẽ lấy 1 leap frame
leap = 1

mtcnn = MTCNN(margin = 20, keep_all=False, select_largest = True, post_process=False, device = device)
# margin nhằm lấy box to hơn, keep_all = False để không lấy hết các output từ ảnh, poss_process = False để giữ cho pixel lưu ko bị normalization về [-1, 1]
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while cap.isOpened() and count:
    isSuccess, frame = cap.read()
    if mtcnn(frame) is not None and leap%2:
        path = str(USR_PATH+'/{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-")+str(count)))
        face_img = mtcnn(frame, save_path = path)
        count-=1
    leap+=1
    cv2.imshow('Face Capturing', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()