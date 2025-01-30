import cv2
import numpy as np
import pickle
import os

def crop_region(image: np.ndarray, corners: np.ndarray, margin: int = 10) -> np.ndarray:
    adjusted = np.array([
        (corners[0][0] - margin, corners[0][1] - margin),
        (corners[1][0] + margin, corners[1][1] - margin),
        (corners[2][0] + margin, corners[2][1] + margin),
        (corners[3][0] - margin, corners[3][1] + margin)
    ], dtype=np.int32)
    
    adjusted[:, 0] = np.clip(adjusted[:, 0], 0, image.shape[1] - 1)
    adjusted[:, 1] = np.clip(adjusted[:, 1], 0, image.shape[0] - 1)
    
    x_min, y_min = np.min(adjusted, axis=0)
    x_max, y_max = np.max(adjusted, axis=0)
    return image[y_min:y_max, x_min:x_max]

def process_single_image(input_path: str, output_path: str, pickle_path: str, margin: int = 10):
    # 파일 존재 확인
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {input_path}")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"피클 파일을 찾을 수 없습니다: {pickle_path}")

    # 피클 파일 로드
    with open(pickle_path, 'rb') as f:
        corners_list = pickle.load(f)
    print(f"corners_list loaded: {corners_list}")

    # 이미지 로드
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {input_path}")
    print(f"Image loaded: shape={image.shape}")

    # output 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cropped_images = []
    for i, corners in enumerate(corners_list):
        cropped = crop_region(image, corners, margin)
        print(f"Cropped image {i} shape: {cropped.shape}")
        cropped_images.append(cropped)

    max_height = max(img.shape[0] for img in cropped_images)
    resized_images = []
    for i, img in enumerate(cropped_images):
        if img.shape[0] != max_height:
            aspect_ratio = img.shape[1] / img.shape[0]
            new_width = int(max_height * aspect_ratio)
            resized = cv2.resize(img, (new_width, max_height))
        else:
            resized = img
        print(f"Resized image {i} shape: {resized.shape}")
        resized_images.append(resized)

    concatenated = np.hstack(resized_images)
    print(f"Concatenated shape: {concatenated.shape}")
    cv2.imwrite(output_path, concatenated)

if __name__ == "__main__":
    # 입력 이미지 경로
    input_path = "/home/gsf/Desktop/kangsan_workspace/gsf_solution/fastapi_server/app/utils/image/image2process/target.png"
    
    """
    피클 파일 정보 예시
    각 꼭짓점의 좌표점을 저장
    [ array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32),  # 첫 번째 스티커
    array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32),  # 두 번째 스티커
    array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)   # 세 번째 스티커 ]
    """
    pickle_path = '/home/gsf/Desktop/kangsan_workspace/gsf_solution/fastapi_server/app/utils/image/image2process/sticker_positions.pkl'
    
    # 출력 이미지 경로
    output_path = "/home/gsf/Desktop/kangsan_workspace/gsf_solution/fastapi_server/app/utils/image/image2process/output/targetCrop.png"
    
    process_single_image(input_path, output_path, pickle_path)