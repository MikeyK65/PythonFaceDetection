


import cv2


def detectFaces(img):
    #grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(img, 1.3, 5, flags = cv2.CASCADE_SCALE_IMAGE)

    if any(faces):
        return img
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

    return img



faceClassifier = cv2.CascadeClassifier ('E:\src\OpenCV\opencv\data\haarcascades\haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")



img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k%256 == 32:
        # SPACE pressed
        im1 = detectFaces(frame)
        cv2.imshow("Video Face Detection", im1)

    elif k%256 == 115:
        # s pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

    elif k%256 == 100:  # Save with face detection
        # d pressed
        im1 = detectFaces(frame)
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, im1)
        print("{} written!".format(img_name))
        img_counter += 1


cam.release()

cv2.destroyAllWindows()



