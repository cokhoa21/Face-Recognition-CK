# CACH DUNG LENH
# python train.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

# import các thư viện cần thiết
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import argparse
import pickle

# khởi tạo các tham số để chạy command line
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# load face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())
# print(data["embeddings"])
# print(data["names"])
# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
# print(labels)

# chia tập data thành 80% tập train và 20% tập test
(trainX, testX, trainY, testY) = train_test_split(data["embeddings"], labels,
	test_size=0.20, stratify=labels, random_state=42)

# train model nhận diện bằng SVM trên các vector embedding face 128 chiều
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(trainX, trainY)

# predict
y_train_predict = recognizer.predict(trainX)
y_test_predict = recognizer.predict(testX)
# score
score_train = accuracy_score(trainY, y_train_predict)
score_test = accuracy_score(testY, y_test_predict)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

# lưu face recognition model
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# lưu label encoder
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()