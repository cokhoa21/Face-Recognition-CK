# CACH DUNG LENH
# python recognize_webcam.py --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle

# import các thư viện cần thiết
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# khởi tạo các tham số để chạy command line
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", type=str, default='face_detector',
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
    help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
    help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
    help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load face detector model từ thư mục
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load face recognition model đã train cùng với label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# khởi tạo videostream, cho phép webcam bật
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# start tính FPS
fps = FPS().start()
# lặp qua các frames từ video stream
while True:
    # đọc frame từ video stream
    frame = vs.read()
    # resize frame về 600 pixel và lấy chiều
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    # tạo 1 blod cho ảnh
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # detect face từ ảnh đầu vào
    detector.setInput(imageBlob)
    detections = detector.forward()
    # lặp qua các detections
    for i in range(0, detections.shape[2]):
        # lấy ra độ tin cậy (xác suất,...) tương ứng của mỗi detection
        confidence = detections[0, 0, i, 2]
        # lọc ra các detections đảm bảo độ tin cậy > ngưỡng tin cậy
        if confidence > args["confidence"]:
            # tính toán (x,y) bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # trích ra face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # đảm bảo chiều rộng và chiều cao của face đủ lớn, nếu nhỏ quá => bỏ qua
            if fW < 20 or fH < 20:
                continue
            # tạo 1 blob cho face ROI, cho blob qua face embedding model để chuyển về vector 128 chiều
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            # cho qua model phân loại để nhận diện mặt
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            # vẽ bounding box quanh face cùng xác suất tương ứng
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    # update FPS counter
    fps.update()
    # show output frame
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1) & 0xFF
    # nếu nhấn 'q' thì thoát
    if key == ord("q"):
        break
# dừng việc tính và hiển thị thông tin FPS
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# cleanup
cv2.destroyAllWindows()
vs.stop()