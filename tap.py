import cv2
import os
import time
import numpy as np
# 選擇攝影機
cap = cv2.VideoCapture(1)
touch_point = []
touch_time = time.time()

ret, frame = cap.read()
tap = False
double_tap = False
long_press = False
double_tap
touched = False
while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # threshold
    ret, thresh1 = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    # open and close
    kernel = np.ones(shape = (5,5))
    thresh1 = cv2.morphologyEx(thresh1,cv2.MORPH_ERODE,kernel,iterations=2)
    
    # find and draw contour
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame,contours,-1,(0,0,255),3)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 3000:
            cX = int(M["m10"] / (M["m00"]))
            cY = int(M["m01"] / (M["m00"]))
            if touched == False:
                touch_time = time.time()
            touched = True

        else:
            if abs(touch_time - time.time()) > 3:
                tap = False
                double_tap = False
                long_press = False
                cv2.imshow('output', black_img)
            touched = False
        if touched:
            if time.time() - touch_time > 1.5:
                long_press = True
                cv2.putText(black_img, str('long_press'), (50, 50),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3)
                cv2.imshow('output', black_img)
            elif tap and abs(touch_time - time.time()) < 0.3:
                double_tap = True
                cv2.putText(black_img, str('double_tap'), (50, 50),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3)
                cv2.imshow('output',black_img)
            elif abs(touch_time - time.time()) > 0.5:
                tap = True
                cv2.putText(black_img, str('tap'), (50, 50),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3)
                cv2.imshow('output', black_img)
            print(double_tap,tap,long_press)
            

	# 顯示圖片

    black_img = np.zeros(shape=frame.shape)

    for i in range(len(touch_point)):
        cv2.circle(black_img, touch_point[i], 7, (255, 255, 255), 10)
        if i > 0:
            cv2.line(black_img,touch_point[i-1],touch_point[i],(255,255,255),5)
    cv2.imshow('gray', np.flip(frame,axis = 1))
    
    # 若按下 q 鍵則離開迴圈
    key = cv2.waitKey(1)
    if key & 0xFF == ord('r'):
        cv2.imwrite('handWrite.jpg',black_img)
        touch_point = []
    if key & 0xFF == ord('q'):
        break


# 釋放攝影機
cap.release()
# out.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
