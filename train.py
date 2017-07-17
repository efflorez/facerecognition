#creating data sets
import cv2, sys, numpy, os
size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
eyes_har = 'haarcascade_eye.xml'
fn_dir = 'data_set' #All the faces data will be present this folder
fn_name = ''
path = 'data_set'
if not os.path.isdir(path):
      os.mkdir(path)
(im_width, im_height) = (112, 92)
haar_cascade = cv2.CascadeClassifier(fn_haar)
eye_cascade = cv2.CascadeClassifier(eyes_har)
webcam = cv2.VideoCapture(0) #'0' is use for my webcam, if you've any other
#camera attached use '1' like this
pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
if n[0]!='.' ]+[0])[-1] + 1
print (pin)
# The program loops until it has 30 images of the face.
count = pin
samples = 0 

while samples < 50:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(gray, (gray.shape[1] / size, gray.shape[0] / size))
    faces = haar_cascade.detectMultiScale(mini)
    faces = sorted(faces, key=lambda x: x[3])
    if faces:
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_color = im[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        eyes = eye_cascade.detectMultiScale(face)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(140,255,0),2)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(im, path, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0)) 
            if len(eyes) > 2 :
 				cv2.imwrite('%s/%s.pgm' % (path, pin+count), face_resize)
				count += 1
				samples += 1
       
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(100)
    if key == 27:
      break