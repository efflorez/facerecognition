# facerec.py
import cv2, sys, numpy, os
import numpy as np
size = 2
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'face_data'


def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)
# Part 1: Create fisherRecognizer
print('Training...')

# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
# Get the folders containing the training data
for (subdirs, dirs, files) in os.walk(fn_dir):
    #print subdirs
    #print dirs
    #print files
    #print 'termino'
    # Loop through each folder named after the subject in the photos
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)

        # Loop through each photo in the folder
        for filename in os.listdir(subjectpath):

            # Skip non-image formates
            f_name, f_extension = os.path.splitext(filename)
            if(f_extension.lower() not in
                    ['.png','.jpg','.jpeg','.gif','.pgm']):
                print( "Skipping " + filename + ", wrong file type" )
                continue
            path = subjectpath + '/' + filename
            lable = id

            # Add to training data
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(im_width, im_height) = (112, 92)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

#model = cv2.createFisherFaceRecognizer()
model = cv2.createEigenFaceRecognizer()
model.train(images, lables)




# Part 2: Use fisherRecognizer on camera stream
haar_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)
timer = 0
while True:

    # Loop until the camera is working
    rval = False
    while(not rval):
        # Put the image from the webcam into 'frame'
        (rval, frame) = webcam.read()
        if(not rval):
            print("Failed to open webcam. Trying again...")

    # Flip the image (optional)
    frame=cv2.flip(frame,1,0)

    # Convert to grayscalel
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    faces = haar_cascade.detectMultiScale(mini)
    for i in range(len(faces)):
        face_i = faces[i]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        #face_resize = cv2.equalizeHist(face_resize)
        # Try to recognize the face
        prediction = model.predict(face_resize)
       
        cara = '%s' % (names[prediction[0]])
        if(prediction[1]/10 < 500):
            
            #if cara == "edgar":
                os.system("bash on.sh")
                cv2.putText(frame,'%s - %.0f' % (names[prediction[0]],prediction[1]/10),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 0))
        else:
                cv2.putText(frame,'Desconocido',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0,0, 255))
                # Apagar Led
                os.system("bash off.sh")
                cv2.imshow('OpenCV', frame)
    cv2.imshow('OpenCV', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
