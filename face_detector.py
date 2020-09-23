import cv2

#load some pre-trained data on face frontals from opencv (haar cascase algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces in
img_name = 'viki.jpg'

#reading image
img = cv2.imread(img_name)

#conversion of image into grayscale image
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect face coordinates
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#displaying detected faces
#here x- x-axis, y- y-axis, w- width and h- height
#face_cordinates is a list

for (x,y,w,h) in face_coordinates:

  # drawing green square on the recognized face
  cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)

  #putText is used to write text onto the image
  #img = cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0, 0, 255),2)

#shows the image stored in img variable
cv2.imshow('face-recognition using OpenCV',img)

#waits until any key is pressed
cv2.waitKey(0)

print("Done")