import cv2 as cv
print(cv.__version__)

face_cascade_db = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

img = cv.imread("./robert/def_scan/img.jpg")

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19)
for (x, y, w, h) in faces:
    cv.rectangle(img, (x,y), (x + w, y+ h) (0, 255, 0), 2)

cv.imshow('rez', img)
cv.waitKey()