# 필요한 라이브러리 임포트
import cv2
import numpy as np
import pickle
import os
from PIL import Image
from typing import List, Tuple, Optional, Any
from pathlib import Path
import sys
import time
from datetime import datetime

def validate_hsv_input(h: int, s: int, v: int) -> bool:
    """HSV 입력값 검증"""
    if not (0 <= h <= 179):
        print(f"Error: H 값은 0-179 사이여야 합니다. 입력값: {h}")
        return False
    if not (0 <= s <= 255):
        print(f"Error: S 값은 0-255 사이여야 합니다. 입력값: {s}")
        return False
    if not (0 <= v <= 255):
        print(f"Error: V 값은 0-255 사이여야 합니다. 입력값: {v}")
        return False
    return True

class ImageProcessor:
    """이미지 처리를 위한 유틸리티 클래스"""
    
    @staticmethod
    def load_and_preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """이미지 로드 및 전처리"""
        image = Image.open(image_path)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        return image_cv, hsv_image

    @staticmethod
    def create_mask(hsv_image: np.ndarray, lower_bound: Tuple[int, int, int], 
                   upper_bound: Tuple[int, int, int]) -> np.ndarray:
        """HSV 이미지에서 마스크 생성"""
        return cv2.inRange(hsv_image, np.array(lower_bound), np.array(upper_bound))

    @staticmethod
    def find_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
        """가장 큰 외곽선 찾기"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea) if contours else None

    @staticmethod
    def find_corners(contour: np.ndarray) -> np.ndarray:
        """외곽선에서 네 꼭짓점 찾기"""
        top_left = min(contour, key=lambda point: point[0][0] + point[0][1])[0]
        top_right = max(contour, key=lambda point: point[0][0] - point[0][1])[0]
        bottom_left = min(contour, key=lambda point: point[0][0] - point[0][1])[0]
        bottom_right = max(contour, key=lambda point: point[0][0] + point[0][1])[0]
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

    @staticmethod
    def apply_margin(corners: np.ndarray, margin: int) -> np.ndarray:
        """꼭짓점에 마진 적용"""
        adjusted = np.array([
            (corners[0][0] - margin, corners[0][1] - margin),  # top_left
            (corners[1][0] + margin, corners[1][1] - margin),  # top_right
            (corners[2][0] + margin, corners[2][1] + margin),  # bottom_right
            (corners[3][0] - margin, corners[3][1] + margin)   # bottom_left
        ], dtype=np.int32)
        return adjusted

    def crop_region(self, image: np.ndarray, corners: np.ndarray, 
                   margin: int = 10, restore_black: bool = True) -> np.ndarray:
        """이미지 영역 크롭"""
        if margin > 0:
            corners = self.apply_margin(corners, margin)

        # 이미지 경계를 벗어나지 않도록 조정
        corners[:, 0] = np.clip(corners[:, 0], 0, image.shape[1] - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, image.shape[0] - 1)

        # 마스크 생성 및 적용
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [corners], 255)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # 크롭 영역 계산
        x_min, y_min = np.min(corners, axis=0)
        x_max, y_max = np.max(corners, axis=0)
        cropped = masked_image[y_min:y_max, x_min:x_max]

        if restore_black:
            # 검은색 영역 복원
            original_crop = image[y_min:y_max, x_min:x_max]
            black_mask = (cropped == 0)
            cropped[black_mask] = original_crop[black_mask]

        return cropped

    def detect_sticker(self, frame: np.ndarray, lower_bound: Tuple[int, int, int], 
                      upper_bound: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """프레임에서 스티커 감지"""
        try:
            # HSV 변환
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 마스크 생성 및 컨투어 찾기
            mask = self.create_mask(hsv_image, lower_bound, upper_bound)
            max_contour = self.find_largest_contour(mask)
            
            if max_contour is None:
                return None

            # 꼭짓점 찾기
            corners = self.find_corners(max_contour)
            return corners

        except Exception as e:
            print(f"Error detecting sticker: {e}")
            return None

class VideoProcessor:
    """비디오 처리를 위한 클래스"""
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.initial_corners_list = []
        self.margin = 10  # 기본 마진값 설정

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭 이벤트 처리 함수"""
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param['frame']
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_value = hsv_frame[y, x]
            print(f"\nHSV 값:")
            print(f"H(색상)={hsv_value[0]} ({hsv_value[0]*2}도)")
            print(f"S(채도)={hsv_value[1]} ({hsv_value[1]/255*100:.1f}%)")
            print(f"V(명도)={hsv_value[2]} ({hsv_value[2]/255*100:.1f}%)")

    def get_hsv_values_from_video(self, video_path: str) -> bool:
        """10프레임 후의 프레임을 보여주며 마우스 클릭으로 HSV 값 확인"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return False

        # 처음 10프레임 건너뛰기
        for _ in range(10):
            ret = cap.read()[0]
            if not ret:
                print("Error: Could not skip frames")
                return False

        # 11번째 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return False
        
        cap.release()

        cv2.namedWindow('Frame for HSV Detection')
        param = {'frame': frame}
        cv2.setMouseCallback('Frame for HSV Detection', lambda event, x, y, flags, param: 
                        self.mouse_callback(event, x, y, flags, param), param)

        print("프레임이 표시됩니다. 스티커의 여러 부분을 클릭하여 HSV 값을 확인하세요.")
        print("종료하려면 'Q'키를 누르세요.")

        while True:
            cv2.imshow('Frame for HSV Detection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        return True

    def visualize_detections(self, frame: np.ndarray, corners_list: List[np.ndarray]) -> np.ndarray:
        """디텍션된 영역을 원본 이미지에 표시"""
        result = frame.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                 (0, 255, 255), (255, 0, 255)]  # 각 스티커별 다른 색상
        
        for i, corners in enumerate(corners_list):
            color = colors[i % len(colors)]
            cv2.polylines(result, [corners], True, color, 2)
            # 번호 표시
            x, y = corners[0][0], corners[0][1]
            cv2.putText(result, str(i+1), (x-10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        return result

    def detect_initial_stickers(self, video_path: str, hsv_params: List[Tuple[Tuple[int, int, int], 
                              Tuple[int, int, int]]]) -> bool:
        """첫 프레임에서 여러 스티커 위치 감지"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return False

        # 처음 10프레임 건너뛰기
        for _ in range(10):
            ret = cap.read()[0]
            if not ret:
                print("Error: Could not skip frames")
                return False
            
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Could not read first frame")
            return False

        self.initial_corners_list = []
        for lower_bound, upper_bound in hsv_params:
            corners = self.image_processor.detect_sticker(frame, lower_bound, upper_bound)
            if corners is None:
                print(f"Error: Could not detect sticker with HSV range {lower_bound} to {upper_bound}")
                return False
            self.initial_corners_list.append(corners)

        # 디텍션 결과 시각화
        result_frame = self.visualize_detections(frame, self.initial_corners_list)
        cv2.imshow('Detected Stickers', result_frame)
        print("\n디텍션 결과를 확인하세요. 계속하려면 아무 키나 누르세요...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return True

    def process_video(self, video_path: str, output_dir: str, frame_interval: int = 1, 
                    duration: float = None, skip_first_n: int = 4) -> None:
        """비디오 처리 및 프레임별 스티커 추출"""
        if not self.initial_corners_list:
            print("Error: Initial sticker positions not detected")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"비디오 FPS: {fps}")

        max_frames = None
        if duration is not None:
            max_frames = int(duration * fps)
            print(f"처리할 프레임 수: {max_frames}")

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames is not None and frame_count >= max_frames):
                break

            if frame_count % frame_interval == 0:
                # 처음 n개의 프레임은 건너뛰기
                if saved_count >= skip_first_n:
                    # 각 스티커 영역 추출
                    cropped_images = []
                    for corners in self.initial_corners_list:
                        cropped = self.image_processor.crop_region(
                            frame,
                            corners,
                            self.margin
                        )
                        cropped_images.append(cropped)

                    # 이미지 가로로 이어붙이기
                    max_height = max(img.shape[0] for img in cropped_images)
                    resized_images = []
                    for img in cropped_images:
                        if img.shape[0] != max_height:
                            aspect_ratio = img.shape[1] / img.shape[0]
                            new_width = int(max_height * aspect_ratio)
                            resized = cv2.resize(img, (new_width, max_height))
                            resized_images.append(resized)
                        else:
                            resized_images.append(img)

                    concatenated = np.hstack(resized_images)
                    
                    # 파일 저장
                    output_path = os.path.join(output_dir, f"frame_{saved_count-skip_first_n:06d}.png")
                    cv2.imwrite(output_path, concatenated)

                saved_count += 1

                if max_frames:
                    progress = (frame_count + 1) / max_frames * 100
                    print(f"\r처리 진행률: {progress:.1f}%", end="")

            frame_count += 1

        cap.release()
        print(f"\n프레임 처리 완료: 총 {frame_count}개 프레임 중 {saved_count-skip_first_n}개 저장됨")


def get_user_parameters() -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """사용자로부터 여러 스티커의 HSV 파라미터 입력받기"""
    hsv_params = []
    
    while True:
        try:
            # 스티커 개수 입력
            sticker_count = int(input("\n처리할 스티커의 개수를 입력하세요: "))
            if sticker_count <= 0:
                print("Error: 스티커 개수는 1개 이상이어야 합니다.")
                continue
            break
        except ValueError:
            print("Error: 올바른 숫자를 입력해주세요.")

    for i in range(sticker_count):
        while True:
            try:
                print(f"\n{i+1}번째 스티커의 파라미터를 입력해주세요:")
                
                # HSV 하한값 입력
                print("\nHSV 하한값 입력:")
                h_min = int(input("H 최소값 (0-179): "))
                s_min = int(input("S 최소값 (0-255): "))
                v_min = int(input("V 최소값 (0-255): "))
                
                if not validate_hsv_input(h_min, s_min, v_min):
                    continue
                    
                # HSV 상한값 입력
                print("\nHSV 상한값 입력:")
                h_max = int(input("H 최대값 (0-179): "))
                s_max = int(input("S 최대값 (0-255): "))
                v_max = int(input("V 최대값 (0-255): "))
                
                if not validate_hsv_input(h_max, s_max, v_max):
                    continue
                
                # 값의 범위 확인
                if h_max < h_min or s_max < s_min or v_max < v_min:
                    print("Error: 상한값은 하한값보다 커야 합니다.")
                    continue
                
                hsv_params.append(((h_min, s_min, v_min), (h_max, s_max, v_max)))
                break
                
            except ValueError:
                print("Error: 올바른 숫자를 입력해주세요.")
    
    return hsv_params

def main():
    # HSV 파라미터와 프레임 설정을 저장할 변수들
    hsv_params = None
    first_run = True

    while True:
        # 비디오 파일 경로 입력
        video_path = input("비디오 파일 경로를 입력하세요: ")
        if not os.path.exists(video_path):
            print("Error: 비디오 파일을 찾을 수 없습니다.")
            continue

        # VideoProcessor 인스턴스 생성 (첫 실행시에만)
        if first_run:
            video_processor = VideoProcessor()
            
            # HSV 값 확인을 위한 비디오 재생
            if not video_processor.get_hsv_values_from_video(video_path):
                continue

            # 사용자로부터 HSV 파라미터 입력받기
            hsv_params = get_user_parameters()
            first_run = False

        # 프레임 간격과 처리 시간 입력
        while True:
            try:
                frame_interval = int(input("\n프레임 추출 간격 (1: 모든 프레임, 2: 2프레임마다, ...): "))
                if frame_interval < 1:
                    print("Error: 프레임 간격은 1 이상이어야 합니다.")
                    continue
                
                duration = float(input("처리할 동영상 길이(초 단위): "))
                if duration <= 0:
                    print("Error: 동영상 길이는 0보다 커야 합니다.")
                    continue
                break
            except ValueError:
                print("Error: 올바른 숫자를 입력해주세요.")

        # 출력 디렉토리 설정
        output_dir = input("\n출력 디렉토리 경로를 입력하세요: ")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not video_processor.initial_corners_list:
            print("\n스티커 위치 감지 중...")
            if not video_processor.detect_initial_stickers(video_path, hsv_params):
                print("Error: 스티커 위치를 감지하지 못했습니다.")
                continue
        
        print("스티커 위치를 감지했습니다. 프레임 추출을 시작합니다...")
        video_processor.process_video(video_path, output_dir, frame_interval, duration)
        print(f"\n처리가 완료되었습니다. 결과물이 {output_dir}에 저장되었습니다.")

        # 추가 처리 여부 확인
        while True:
            response = input("\n다음 동영상을 추가로 저장하시겠습니까? (C: 계속, Q: 종료): ").upper()
            if response in ['C', 'Q']:
                break
            print("잘못된 입력입니다. C 또는 Q를 입력해주세요.")

        if response == 'Q':
            break

def make_pickle():
    video_path = "/home/gsf/Desktop/kangsan_workspace/gsf_solution/fastapi_server/app/utils/image/image2process/nomal.mp4"
    
    # HSV 파라미터 직접 설정
    hsv_params = [
        ((160, 140, 135), (178, 160, 155)),  # 주황색
        ((145, 145, 60), (175, 185, 85)),    # 빨강
        ((130, 185, 60), (160, 205, 85))     # 보라
    ]

    video_processor = VideoProcessor()
    
    print("\n스티커 위치 감지 중...")
    if video_processor.detect_initial_stickers(video_path, hsv_params):
        # corners_list 피클파일로 저장
        with open('sticker_positions.pkl', 'wb') as f:
            pickle.dump(video_processor.initial_corners_list, f)
        print("스티커 위치가 sticker_positions.pkl에 저장되었습니다.")
        
if __name__ == "__main__":
    # main()
    make_pickle()