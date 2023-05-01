import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence = 0.8 , min_tracking_confidence = 0.5 )

tipIds = [4, 8, 12, 16, 20]


def drawHandLandmarks(image, hand_landmarks):
    print("--------------")
    if hand_landmarks:
        for l in hand_landmarks:
            mp_drawing.draw_landmarks(image , l , mp_hands.HAND_CONNECTIONS)


def countFingers(image, hand_landmarks, handNo=0):
    if hand_landmarks:
        landmarks = hand_landmarks[handNo].landmark

        fingers = []

        for index in tipIds :
            fingerTipY = landmarks[index].y
            fingerBottomY = landmarks[index - 2].y

            if index != 4:
                if fingerTipY < fingerBottomY:
                    fingers.append(1)

                if fingerTipY > fingerBottomY:
                    fingers.append(0)

                
    totalFingers = fingers.count(1)

    text = f'Fingers: {totalFingers}'
    
    cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)


while True:
    success, image = cap.read()

    image = cv2.flip(image , 1)

    results = hands.process(image)

    handLandmarks = results.multi_hand_landmarks
    
    drawHandLandmarks(image, handLandmarks)

    countFingers(image , handLandmarks)

    cv2.imshow("Media Controller", image)

    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()

