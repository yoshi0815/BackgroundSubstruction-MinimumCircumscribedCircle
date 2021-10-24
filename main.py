# -*- coding: utf-8 -*-

#read libraly
import cv2
import numpy as np
import time

# 画像の読み込み
img_src1 = cv2.imread("a.jpg", 1)
img_src2 = cv2.imread("b.jpg", 1)

#アルゴリズム
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv2.createBackgroundSubtractorMOG()

fgmask = fgbg.apply(img_src1)
fgmask = fgbg.apply(img_src2)

# 表示
cv2.imshow('frame',fgmask)

# 検出画像
bg_diff_path  = './diff.jpg'
cv2.imwrite(bg_diff_path,fgmask)

time.sleep(1)

frame = cv2.imread("diff.jpg", 1)
#cap = cv2.VideoCapture(0)

#hsv変換
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#white_hsv値域1
hsv_min = np.array([0, 0, 100])
hsv_max = np.array([180, 45, 255])
mask1   = cv2.inRange(hsv, hsv_min, hsv_max)

#赤色以外マスク処理
res_red = cv2.bitwise_and(frame, frame, mask=mask1)
#res_red2 = cv2.bitwise_and(frame, frame, mask=mask2)
sum_red = res_red #+ res_red2
#cv2.imshow('mask', sum_red)

#輪郭取得
gray = cv2.cvtColor(sum_red, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray', gray)
ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
#cv2.imshow('thresh', thresh)
img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#輪郭描画
cv2.drawContours(frame, contours, -1, (0,255,0), 1)
#cv2.imshow('rinksku', frame)

#一番大きい輪郭検出
contours.sort(key=cv2.contourArea,reverse=True)

cv2.drawContours(frame, contours, 0, (255,0,0), 3)
#cv2.imshow('rinksku_big', frame)

if len(contours) > 0:
    cnt = contours[0]

    #最小外接円描く
    (x,y), radius = cv2.minEnclosingCircle(cnt)
    center        = (int(x), int(y))
    radius        = int(radius)
    cv2.circle(frame, center, radius, (0,255,0), 2)
    print(x,y)

#cv2.imshow('video', frame)
bg_diff_circlepath  = './diff_circle.jpg'
cv2.imwrite(bg_diff_circlepath,frame)

cv2.destroyAllWindows()
cv2.waitKey(0)
