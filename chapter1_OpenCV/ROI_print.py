import cv2 as cv
import sys
import numpy as np

img = cv.imread('soccer.jpg') 

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

clone = img.copy()  
roi = None
start_x, start_y, end_x, end_y = -1, -1, -1, -1 
drawing = False

def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, drawing, roi, img

    if event == cv.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        drawing = True  # 드래그 시작

    elif event == cv.EVENT_MOUSEMOVE: 
        if drawing: 
            temp_img = img.copy()
            cv.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv.imshow('Image', temp_img)

    elif event == cv.EVENT_LBUTTONUP:  #마우스 버튼을 놓았을 때
        end_x, end_y = x, y
        drawing = False  #드래그 종료

        x1, y1, x2, y2 = min(start_x, end_x), min(start_y, end_y), max(start_x, end_x), max(start_y, end_y)

        # ROI 추출
        roi = clone[y1:y2, x1:x2].copy()
        
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.imshow('Image', img)

        if roi.size > 0:
            cv.imshow('ROI_image', roi)

cv.namedWindow('Image')
cv.setMouseCallback('Image', draw_rectangle)
cv.imshow('Image', img)

while True:
    key = cv.waitKey(1)

    if key == ord('r'):
        img = clone.copy()
        roi = None
        cv.imshow('Image', img)
        cv.destroyWindow('ROI_image') 

    elif key == ord('s') and roi is not None:
        cv.imwrite('ROI_image.jpg', roi)
        print('이미지가 저장되었습니다.')

    elif key == ord('q'):  # 종료
        break

cv.destroyAllWindows()
