from deepface import DeepFace
import cv2
import face_recognition

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    demography = DeepFace.analyze(img)
    print("Age: ", demography["age"])
    print("Gender: ", demography["gender"])
    print("Emotion: ", demography["dominant_emotion"])
    print("Race: ", demography["dominant_race"])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceLoc = face_recognition.face_locations(img)[0]
    encodeMark = face_recognition.face_encodings(img)[0]
    y1, x2, y2, x1 = faceLoc
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, demography["dominant_emotion"], (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('webcam', img)
    c = cv2.waitKey(1) & 0xFF

    if c == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()