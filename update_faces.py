# CACH DUNG LENH
# python update_faces.py --dataset data/ --embeddings output/embeddings.pickle --embedding-model openface_nn4.small2.v1.t7

from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# Cac tham so dau vao
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to input dataset")
ap.add_argument("-d", "--detector", type=str, default='face_detector',
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-m", "--embedding-model", required=True,
    help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-e", "--embeddings", required=True,
    help="path to output serialized db of facial embeddings")
ap.add_argument("-s", "--skip", type=int, default=1,
                help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# Load model ssd nhan dien mat
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
# lay duong dan chua anh
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
# khoi tao list chứa face embedding va list name tuong ung
knownEmbeddings = []
knownNames = []
# khoi tao tong so face da xu li
total = 0
knownTotal = []

# Doc file anh va xu li
for (i, imagePath) in enumerate(imagePaths):
    # lay ra name tu duong dan imagePath
    print("[INFO] processing image {}/{}".format(i + 1,
                                             len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    name = name.split('/')[0]

    # load image, resize ve 600 pixel va lay chieu(dimensions)
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # tao 1 blod tu image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

    # Phat hien cac khuon mat trong image
    detector.setInput(blob)
    detections = detector.forward()

    # Neu tim thay it nhat 1 khuon mat
    if len(detections) > 0:
        # Tim khuon mat to nhat trong anh
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # Neu muc do nhan dien > threshold
        if confidence > args["confidence"]:
            # Tach khuon mat va ghi ra file
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # trich ra face ROI va lay cac chieu(dimensions)
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # dam bao chieu rong va chieu cao cua face du lon, neu nho qua => bo qua
            if fW < 20 or fH < 20:
                continue

            # tao 1 blob cho face ROI, cho blob qua face embedding model de chuyen ve vector 128 chieu
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                     (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            # them name va face embedding tuong ung vao cac list da khoi tao
            knownNames.append(name)
            if(knownNames[total] != knownNames[total-1]):
                print(knownNames[total-1], total)
            knownEmbeddings.append(vec.flatten())
            total += 1

# luu lai face embeddings va name ra file
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
print(data)
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()