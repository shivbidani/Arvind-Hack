from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from math import sqrt
import csv
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def dist(x1, y1, x2, y2):
	return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def abs_diff(p1, p2):
	return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])


def mid_pt(p1, p2):
	return (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2


def feature_extraction(path):

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	image = cv2.imread(path)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = detector(gray, 1)
	for (i, face) in enumerate(faces):

		shape = predictor(gray, face)
		shape = face_utils.shape_to_np(shape)
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(face)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
	
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
 
		# show the output image with the face detections + facial landmarks
		#cv2.imshow("Output", image)
		cv2.waitKey(0)
		pts = {"1": 0, "15": 16, "4": 3, "12": 13, "8": 8, "67": 66, "28": 36, "33": 45, "30": 39, "35": 42, "22": 17,
			   "29": 37, "25": 21, "19": 22, "34": 43, "16": 26, "26": 20, "27": 18, "20": 23, "21": 25}

		# Calculation of parameters
		chk_w = dist(shape[pts["1"]][0], shape[pts["1"]][1], shape[pts["15"]][0], shape[pts["15"]][1])
		jaw_w = dist(shape[pts["4"]][0], shape[pts["4"]][1], shape[pts["12"]][0], shape[pts["12"]][1])
		cjwr = chk_w / jaw_w
		color = gray[shape[30][1], shape[30][0]]


		N1 = mid_pt(shape[pts["29"]], shape[pts["34"]])
		N3 = shape[20]
		N4 = shape[23]

		m1 = (shape[pts["28"]][1] - N3[1]) / (shape[pts["28"]][0] - N3[0])
		m2 = (shape[pts["33"]][1] - N4[1]) / (shape[pts["33"]][0] - N4[0])
		x_temp = int((N3[1] - N4[1] - m1 * N3[0] + m2 * N4[0]) / (m2 - m1))
		y_temp = int(N3[1] + m1 * (x_temp - N3[0]))
		N2 = (x_temp, y_temp)

		ufc_h = dist(shape[pts["67"]][0], shape[pts["67"]][1], N1[0], N1[1])
		whr = chk_w / ufc_h

		perimeter = cv2.arcLength(np.array(
			[shape[pts["1"]], shape[pts["4"]], shape[pts["8"]], shape[pts["12"]], shape[pts["15"]], shape[pts["1"]]]), True)
		area = cv2.contourArea(np.array(
			[shape[pts["1"]], shape[pts["4"]], shape[pts["8"]], shape[pts["12"]], shape[pts["15"]], shape[pts["1"]]]))
		par = perimeter / area

		es = 0.5 * (dist(shape[pts["28"]][0], shape[pts["28"]][1], shape[pts["33"]][0], shape[pts["33"]][1]) -
					(dist(shape[pts["30"]][0], shape[pts["30"]][1], shape[pts["35"]][0], shape[pts["35"]][1])))

		lfh = (shape[pts["8"]][1] - shape[pts["1"]][1])
		lffh = lfh / dist(N2[0], N2[1], shape[pts["8"]][0], shape[pts["8"]][1])

		fwlfh = chk_w / lfh

		dist1 = dist(shape[pts["22"]][0], shape[pts["22"]][1], shape[pts["28"]][0], shape[pts["28"]][1])
		dist2 = dist(shape[pts["29"]][0], shape[pts["29"]][1], N3[0], N3[1])
		dist3 = dist(shape[pts["25"]][0], shape[pts["25"]][1], shape[pts["30"]][0], shape[pts["30"]][1])
		dist4 = dist(shape[pts["19"]][0], shape[pts["19"]][1], shape[pts["35"]][0], shape[pts["35"]][1])
		dist5 = dist(shape[pts["34"]][0], shape[pts["34"]][1], N4[0], N4[1])
		dist6 = dist(shape[pts["16"]][0], shape[pts["16"]][1], shape[pts["33"]][0], shape[pts["33"]][1])
		meh = (dist1 + dist2 + dist3 + dist4 + dist5 + dist6)/6
		
		features = {"cjwr": cjwr, "whr": whr, "par": par, "es": es, "lffh": lffh, "fwlfh": fwlfh, "meh": meh, "col":color}
		return features


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values

    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def main():
	path = "/home/shiv/ML/facial-landmarks/images/"
	#ar = []
	#for i in range(0,len(var)):
	#	temp_path = path+var[i]['path']
	#features=feature_extraction(path)
	final_result = []
	with open("/home/shiv/ML/facial-landmarks/dataset_mugshots.csv") as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			try:
				features = feature_extraction(path+ row['path'])
				final_result.append(merge_two_dicts(row, features))
			except Exception as e:
				print(e)
			print('processed:' + row['path'])


	print(final_result)

	keys = final_result[0].keys()

	with open("/home/shiv/ML/facial-landmarks/final_results.csv", "w") as final:
		writer = csv.DictWriter(final, fieldnames=keys)
		writer.writeheader()
		writer.writerows(final_result)


			


if __name__ == "__main__":
	main()
