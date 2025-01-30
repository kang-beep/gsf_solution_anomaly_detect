import cv2
import numpy as np
import os
from PIL import Image
from typing import Tuple, Optional, List, Any

class ImageProcessor:
    """이미지 처리를 위한 유틸리티 클래스"""
    
    @staticmethod
    def load_and_preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """이미지 로드 및 전처리"""
        image = Image.open(image_path)
        # image = image.resize(resize_dim, Image.LANCZOS)
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
        # points = contour.reshape(-1, 2)
        # top_left = points[np.argmin(points.sum(axis=1))]
        # top_right = points[np.argmax(points[:, 0] - points[:, 1])]
        # bottom_left = points[np.argmin(points[:, 0] - points[:, 1])]
        # bottom_right = points[np.argmax(points.sum(axis=1))]
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

    @staticmethod
    def draw_corners_and_edges(image: np.ndarray, corners: np.ndarray, 
                             color: Tuple[int, int, int] = (255, 0, 100), 
                             thickness: int = 2) -> None:
        """꼭짓점과 엣지 그리기"""
        for i in range(4):
            cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), color, thickness)

    def crop_region(self, image: np.ndarray, corners: np.ndarray, 
                   margin: int = 0, restore_black: bool = True) -> np.ndarray:
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

    def detect_stickers(self, image_path: str, lower_bound: Tuple[int, int, int], 
                       upper_bound: Tuple[int, int, int], margin: int = 0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """스티커 감지 및 처리"""
        try:
            # 이미지 로드 및 전처리
            image_cv, hsv_image = self.load_and_preprocess_image(image_path=image_path)
            
            # 마스크 생성 및 컨투어 찾기
            mask = self.create_mask(hsv_image, lower_bound, upper_bound)
            max_contour = self.find_largest_contour(mask)
            
            if max_contour is None:
                return None, None

            # 꼭짓점 찾기
            corners = self.find_corners(max_contour)
            
            # 결과 이미지 생성
            result_image = image_cv.copy()
            self.draw_corners_and_edges(result_image, corners)
            if margin > 0:
                adjusted_corners = self.apply_margin(corners, margin)
                self.draw_corners_and_edges(result_image, adjusted_corners, color=(0, 255, 0))
            
            return corners, result_image

        except Exception as e:
            print(f"Error detecting stickers: {e}")
            return None, None

class BatchProcessor:
    """일괄 처리를 위한 클래스"""
    def __init__(self):
        self.processor = ImageProcessor()
        self.contours_list = [] # 초기화 추가
        self.margin_list = []   # 초기화 추가

    def process_image_batch(self, image_path: str, table_data: List[dict]) -> dict:
        """이미지 일괄 처리"""
        processed_images = []
        detection_failures = []
        self.contours_list = []
        self.margin_list = []
        image_data = []

        for index, item in enumerate(table_data):
            corners, _ = self.processor.detect_stickers(
                image_path,
                (item["lower_bound"]["h"], item["lower_bound"]["s"], item["lower_bound"]["v"]),
                (item["upper_bound"]["h"], item["upper_bound"]["s"], item["upper_bound"]["v"])
            )

            if corners is not None:
                self.contours_list.append(corners)
                self.margin_list.append(item["margin"])
                
                cropped = self.processor.crop_region(
                    cv2.imread(image_path),
                    corners,
                    item["margin"]
                )

                # 이미지 데이터를 PNG 형식으로 인코딩
                _, img_encoded = cv2.imencode('.png', cropped)
                image_data.append((f"crop_image{index+1}.png", img_encoded.tobytes()))
                processed_images.append(f"crop_image{index+1}.png")
            else:
                detection_failures.append(index + 1)

        # 전체 결과 이미지 생성
        if self.contours_list:
            image_cv = cv2.imread(image_path)
            result_image = image_cv.copy()
            for corners, margin in zip(self.contours_list, self.margin_list):
                adjusted_corners = self.processor.apply_margin(corners, margin)
                self.processor.draw_corners_and_edges(result_image, adjusted_corners, color=(0, 255, 0))
            
            # 전체 결과 이미지 인코딩
            _, all_crop_encoded = cv2.imencode('.png', result_image)
            image_data.append(("all_crop.png", all_crop_encoded.tobytes()))

        return {
            "processed_images": processed_images,
            "failures": detection_failures,
            "contours": self.contours_list,
            "margins": self.margin_list,
            "image_data": image_data,
        }
