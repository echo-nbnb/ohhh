"""测试 ColorStabilityDetector — 独立运行"""
import cv2
from color_stability_detector import ColorStabilityDetector

detector = ColorStabilityDetector(roi_size=40, confirm_seconds=3.0)
detector.start()
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)

    result = detector.update(frame)
    if result:
        print(f"[确认] {result}")

    detector.draw_roi(frame)
    cv2.imshow("Color Detector", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
