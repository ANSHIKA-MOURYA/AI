import cv2
import mediapipe as mp

# Initialize MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7, model_complexity=1)
mpDraw = mp.solutions.drawing_utils

# OpenCV camera
cap = cv2.VideoCapture(0)

# App state
step = 0
operation = None
result = None

# Operation button zones (x1, y1, x2, y2)
operations = {
    'ADD': (50, 50, 150, 100),
    'SUB': (200, 50, 300, 100),
    'MUL': (350, 50, 450, 100),
    'DIV': (500, 50, 600, 100),
}

# Count fingers function
def count_fingers(hand_landmarks, hand_label):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_label == "Right":
        if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0]-1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0]-1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Draw operation buttons
    for op, (x1, y1, x2, y2) in operations.items():
        color = (255, 0, 0) if op == operation else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, op, (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Detect hands
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if step == 0:
        cv2.putText(frame, "Select an Operation", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        if results.multi_hand_landmarks:
            x = int(results.multi_hand_landmarks[0].landmark[8].x * w)
            y = int(results.multi_hand_landmarks[0].landmark[8].y * h)
            for op, (x1, y1, x2, y2) in operations.items():
                if x1 < x < x2 and y1 < y < y2:
                    operation = op
                    step = 1
                    break

    elif step == 1:
        cv2.putText(frame, f"Selected: {operation}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        cv2.putText(frame, "Show 2 hands to calculate", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            try:
                num1, num2 = 0, 0
                for i, handLms in enumerate(results.multi_hand_landmarks):
                    hand_label = results.multi_handedness[i].classification[0].label
                    if i == 0:
                        num1 = count_fingers(handLms, hand_label)
                    else:
                        num2 = count_fingers(handLms, hand_label)

                if operation == 'ADD':
                    result = num1 + num2
                elif operation == 'SUB':
                    result = num1 - num2
                elif operation == 'MUL':
                    result = num1 * num2
                elif operation == 'DIV':
                    result = round(num1 / num2, 2) if num2 != 0 else "Error"

                step = 2
            except:
                pass

    elif step == 2:
        cv2.putText(frame, f"{operation}: {result}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 128, 255), 4)
        cv2.putText(frame, "Press R to reset", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Finger Calculator", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        step = 0
        operation = None
        result = None

cap.release()
cv2.destroyAllWindows()
