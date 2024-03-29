import cv2
import time
import os
import HandTrackingModule as htm

# Parameters____________________
capWidth, capHeight = 640, 480
# ______________________________

cap = cv2.VideoCapture(0)

# Changing Resolution
cap.set(3, capWidth)
cap.set(4, capHeight)

# Storing Finger Count images
folderPath = "Finger_images"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# For measuring fps
prev_time = 0
curr_time = 0

# Creating Object for Hand Detection Module
detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findposition(img, draw=False)

    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:  # Checking if Thumb is right from middle of a thumb
            fingers.append(1)
        else:
            fingers.append(0)
        # Other Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:  # Checking if Tip is above from middle of a finger
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        print(totalFingers)

        # Image Slicing for putting overlay images
        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]

        # Showing Number of fingers Open
        cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 387), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, f'fps: {(int(fps))}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()
