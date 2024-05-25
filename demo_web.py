import argparse
import base64
import os
import pickle
import time

import cv2
import imutils
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for
from imutils.video import FPS
from imutils.video import VideoStream
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['detector_model'] = 'face_detector'
app.config['embedding_model'] = 'openface_nn4.small2.v1.t7'
app.config['recognizer'] = 'output/recognizer.pickle'
app.config['le'] = 'output/le.pickle'
app.config['confidence'] = 0.5


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


def gen():
    # load face detector model từ thư mục
    print("[INFO] loading face detector...")
    protoPath = os.path.join(app.config['detector_model'], 'deploy.prototxt')
    modelPath = os.path.join(app.config['detector_model'], 'res10_300x300_ssd_iter_140000.caffemodel')
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load serialized face embedding model
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(app.config['embedding_model'])

    # load face recognition model đã train cùng với label encoder
    recognizer = pickle.loads(open(app.config['recognizer'], "rb").read())
    le = pickle.loads(open(app.config['le'], "rb").read())
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
            if confidence > app.config['confidence']:
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


@app.route('/webcam', methods=['GET'])
def webcam():
    Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return redirect(url_for('index'))



app.config['image_upload'] = 'test_images'


@app.route('/upload', methods=['GET', 'POST'])
def input_image():
    if request.method == 'POST':
        image = request.files['image']
        print(type(image))
        filename = secure_filename(image.filename)
        print(filename)
        image.save(os.path.join(app.config['image_upload'], filename))
        return redirect(url_for('predict', filename=filename))
    return redirect(request.url)


@app.route('/predict_<filename>')
def predict(filename):

    img_path = app.config['image_upload'] + '/' + filename
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # tạo 1 blod cho ảnh
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # load face detector model từ thư mục
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([app.config['detector_model'], "deploy.prototxt"])
    modelPath = os.path.sep.join([app.config['detector_model'], "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load serialized face embedding model
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(app.config['embedding_model'])
    recognizer = pickle.loads(open(app.config['recognizer'], "rb").read())
    le = pickle.loads(open(app.config['le'], "rb").read())


    # detect face từ ảnh đầu vào
    detector.setInput(imageBlob)
    detections = detector.forward()

    # lặp qua các detections
    for i in range(0, detections.shape[2]):
        # lấy ra độ tin cậy (xác suất,...) tương ứng của mỗi detection
        confidence = detections[0, 0, i, 2]
        # lọc ra các detections đảm bảo độ tin cậy > ngưỡng tin cậy
        if confidence > app.config['confidence']:
            # tính toán (x,y) bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # trích ra face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # đảm bảo chiều rộng và chiều cao của face đủ lớn, nếu nhỏ quá => bỏ qua
            if fW < 20 or fH < 20:
                continue
        # tạo 1 blob cho face ROI, cho blob qua face embedding model để chuyển về vector 128 chiều
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                         (0, 0, 0), swapRB=True, crop=False)
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
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    image_content = cv2.imencode('.jpg', image)[1].tobytes()
    encoded_image = base64.encodebytes(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return render_template('index.html', image_to_show=to_send, init=True)


if __name__ == '__main__':
    app.run(debug=True)