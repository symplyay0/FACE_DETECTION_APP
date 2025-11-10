import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze emotion
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']

    # Display on screen
    cv2.putText(frame, f'Mood: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()