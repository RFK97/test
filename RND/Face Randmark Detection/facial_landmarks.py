
# coding: utf-8

# In[7]:


# face_utils.py의 두 함수
#1. rect_to_bb(rect) -> opencv는 x,y,width,height로 생각하므로 rect 객체를 가져와서 좌표의 4튜플로 변환
#2. shape_to_np(shape, dtype="int"):-> 68개의 x,y 좌표를 포함하는 모양 객체 반환


# In[6]:


from imutils import face_utils #근데 밑에서 가져올거면 위에서 왜가져오냐..
import numpy as np
import argparse #명령행 인지 파싱모듈
import imutils #이미지 좌우반전, 회전, 색상변환, 토폴로지골격화등을 해주는 함수가 담긴모듈 https://www.pyimagesearch.com/2015/02/02/just-open-sourced-personal-imutils-package-series-opencv-convenience-functions/ 
import dlib
import cv2


# In[7]:


#argparse는 아직 명확히 이해안감.. 돌려보고싶은데 잘 안돌아가서..
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor" , required = True, 
                help="path to facial landmark predictor")
ap.add_argument("-i","--image",required=True,
               help="path to input image")
args=vars(ap.parse_args()) #이줄 이해안감


# In[8]:


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"]) #그래서 이줄도 이해안감
#원래라면 args=ap.parse_args() 해서 args.shape_predictor 를 하는데 vars함수때문에 바뀐듯한데..


# In[ ]:


#load images, resize, conver to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# detect faces in the grayscale image
rects = detector(gray, 1)


# In[5]:


for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
 
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
 
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)

