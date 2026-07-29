import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return 360 - angle if angle > 180 else angle

model_path = r'C:\Users\kr070\Desktop\Krishna\pose_landmarker.task'

base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options)
detector = vision.PoseLandmarker.create_from_options(options)

# Try camera indices 0 and 1
cap = None
for idx in [0, 1, 2]:
    test = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if test.isOpened():
        print(f"✅ Webcam found at index {idx}")
        cap = test
        break
    test.release()

if cap is None:
    print("❌ No webcam found! Check if another app is using it.")
    exit()

count = 0
stage = None

print("Starting... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Lost webcam feed!")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    h, w = frame.shape[:2]
    angle = None

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        def get_point(idx):
            l = landmarks[idx]
            return [l.x * w, l.y * h]

        try:
            l_vis = landmarks[25].visibility
            r_vis = landmarks[26].visibility

            if l_vis >= r_vis:
                hip, knee, ankle = get_point(23), get_point(25), get_point(27)
            else:
                hip, knee, ankle = get_point(24), get_point(26), get_point(28)

            angle = calculate_angle(hip, knee, ankle)

            for pt in [hip, knee, ankle]:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 10, (0, 200, 255), -1)
            cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (0,200,255), 3)
            cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankle[0]), int(ankle[1])), (0,200,255), 3)
            cv2.putText(frame, f"{int(angle)}", (int(knee[0])+15, int(knee[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

            if angle > 150 and stage != "straight":
                if stage == "bent":
                    count += 1
                    print(f"✅ Rep counted! Total: {count}")
                stage = "straight"
            elif angle < 110 and stage == "straight":
                stage = "bent"

        except Exception as e:
            print(f"Landmark error: {e}")

        for lmk in landmarks:
            cx, cy = int(lmk.x * w), int(lmk.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 120), -1)

    cv2.rectangle(frame, (0, 0), (280, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Reps:  {count}", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 120), 2)
    cv2.putText(frame, f"Stage: {stage or '-'}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, f"Angle: {int(angle) if angle else 'no pose'}", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Sit-up Counter", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal count: {count} reps")