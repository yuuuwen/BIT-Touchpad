import os
import time
import cv2
import joblib
from sklearn.datasets import fetch_openml
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import numpy as np
# 選擇攝影機

def train():
	print("Training")
	dataset = fetch_openml('mnist_784')
	features = np.array(dataset.data, 'int16') 
	labels = np.array(dataset.target, 'int')
	list_hog_fd = []
	for feature in features:
		fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
		list_hog_fd.append(fd)
	hog_features = np.array(list_hog_fd, 'float64')
	pp = preprocessing.StandardScaler().fit(hog_features)
	hog_features = pp.transform(hog_features)
	# Create an linear SVM object
	clf = LinearSVC()

	# Perform the training
	clf.fit(hog_features, labels)

	# Save the classifier
	joblib.dump((clf, pp), "digits_cls.pkl", compress=3)
	print('Training Done')

def getNumber(im):
    im = cv2.dilate(im.copy(),np.ones(shape = (3,3)),iterations = 5)
    clf, pp = joblib.load("digits_cls.pkl")

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.GaussianBlur(im_gray, (5, 5), 0)
    im_display = im.copy()
    ctrs, hier = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(im_display, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 255), 3) 
        # Make the rectangular region around the digit
        
        leng = int(rect[3] * 1)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im[pt1:pt1+leng, pt2:pt2+leng]		

        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_CUBIC)
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
        roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
        nbr = clf.predict(roi_hog_fd)
        cv2.putText(im_display, str(int(nbr[0])), (rect[0]-50, rect[1]+50),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3)

        return im_display






cap = cv2.VideoCapture(1)
touch_point = []
touch_time = time.time()

ret, frame = cap.read()
while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # threshold
    ret, thresh1 = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    # open and close
    kernel = np.ones(shape = (5,5))
    thresh1 = cv2.morphologyEx(thresh1,cv2.MORPH_ERODE,kernel,iterations=3)
    
    # find and draw contour
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame,contours,-1,(0,0,255),3)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 1800:
            cX = int(M["m10"] / (M["m00"]))
            cY = int(M["m01"] / (M["m00"]))
            if time.time() - touch_time > 0.5:
                touch_time = time.time()
                touch_point.append((cX, cY))

            for point in touch_point:
                cv2.circle(frame, point, 7, (0, 0, 0), 10)

	# 顯示圖片
    black_img = np.zeros(shape=frame.shape)

    for i in range(len(touch_point)):
        if i > 0:
            cv2.line(black_img,touch_point[i-1],touch_point[i],(255,255,255),10)
    cv2.imshow('output', np.flip(black_img,axis = 1))

    cv2.imshow('gray', np.flip(frame,axis = 1))
    
    # 若按下 q 鍵則離開迴圈
    key = cv2.waitKey(1)
    if key & 0xFF == ord('r'):
        cv2.imwrite('handWrite.jpg',np.flip(black_img,axis = 1))
        im = cv2.imread('handWrite.jpg')
        black_img = getNumber(im)
        cv2.imshow('output', black_img)
        cv2.waitKey()
        touch_point = []
    if key & 0xFF == ord('q'):
        break


# 釋放攝影機
cap.release()
# out.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
