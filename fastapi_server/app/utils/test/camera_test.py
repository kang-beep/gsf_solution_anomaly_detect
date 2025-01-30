import cv2

cap = cv2.VideoCapture(2)

# 지원되는 프레임 레이트 확인
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"카메라 프레임 레이트: {fps}fps")

# 프레임 레이트 설정 (예: 30fps로 설정)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()