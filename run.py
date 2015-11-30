import pickle
import cv2
import stasm
from itertools import combinations
from scipy.spatial import distance
import numpy as np
from sklearn import preprocessing

model = pickle.load(open("rbf_model.bin", "rb"))

img_neutral = cv2.imread("test/neutral.png", cv2.IMREAD_GRAYSCALE)
img_emotion = cv2.imread("test/emotion.png", cv2.IMREAD_GRAYSCALE)


landmarks = stasm.search_single(img_neutral)
landmarks = stasm.force_points_into_image(landmarks, img_neutral)
neutral_list = []
for point in landmarks:
	neutral_list.append([point[0], point[1]])
neutral_dists = {}
for x,y in combinations(neutral_list, 2):
	header = str(neutral_list.index(x)) + ":" + str(neutral_list.index(y))
	neutral_dists[header] = distance.euclidean(np.array(x), np.array(y))

landmarks = stasm.search_single(img_emotion)
landmarks = stasm.force_points_into_image(landmarks, img_emotion)
emotion_list = []
for point in landmarks:
	emotion_list.append([point[0], point[1]])
emotion_dists = {}
for x,y in combinations(emotion_list, 2):
	header = str(emotion_list.index(x)) + ":" + str(emotion_list.index(y))
	emotion_dists[header] = distance.euclidean(np.array(x), np.array(y))

final_dists = {}
for key in neutral_dists.keys():
	final_dists[key] = 100*(emotion_dists[key]-neutral_dists[key])/neutral_dists[key]

np_points = np.matrix(final_dists.values())
np_points = np.nan_to_num(np_points)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(np_points)

pca = pickle.load(open("pca.bin", "rb"))
pca_features = pca.transform(x_scaled)
print pca_features
print model.predict(pca_features)
