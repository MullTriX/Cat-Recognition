import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalcatface_extended.xml")
'''
# Cat recognition from picture

img=cv2.imread("rusek.jpg")

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=3, minSize=(75,75))
print(f"Number of detected cats: {len(faces)}")
# print(type(faces))
# print(faces)
if len(faces) > 0:
    print("Cat face detected")
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w ,y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x ,y - 18), (x + w, y), (0, 255, 0), -1)
        cv2.putText(img, "Rusek jebany", (x + 45, y - 3),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow("cat face",img)
cv2.waitKey(0)

'''

# Cat recognition form Cameraś
cap = cv2.VideoCapture(0)

while 1:  

    # reads frames from a camera  
    ret, img = cap.read()  
  
    # convert to gray scale of each frames  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  
    # Detects faces of different sizes in the input image  
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  
  
    for (x,y,w,h) in faces:  
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w]  
        roi_color = img[y:y+h, x:x+w]  
  
  
    # Display an image in a window  
    cv2.imshow('img',img)  
  
    # Wait for Esc key to stop  
    k = cv2.waitKey(30) & 0xff
    if k == 27:  
        break
  
# Close the window  
cap.release()  
# De-allocate any associated memory usage  
cv2.destroyAllWindows()
