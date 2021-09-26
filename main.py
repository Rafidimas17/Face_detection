import cv2
cam=cv2.VideoCapture(0)
faceDetector=cv2.CascadeClassifier('haarcascade.xml')
eyesDetector=cv2.CascadeClassifier('haarcascade_eye.xml')
while True:
    ret, img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetector.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_img=img[y:y+h,x:x+w]
        eyes=eyesDetector.detectMultiScale(roi_gray, 1.3, 5)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_img,(ex,ey), (ex+ew, ey+eh),(200,0,0),2)
    cv2.imshow("webcam", img)
    k=cv2.waitKey(1)&0xFF
    if k==27 or k==ord('x'):
        break
cam.release()
cv2.destroyAllWindows()