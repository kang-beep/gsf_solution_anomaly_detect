import cv2
import numpy as np
from ultralytics import YOLO
from math import gcd

class BrightnessProcessor:
    def __init__(self, model_path):
        """
        초기화 함수
        Args:
            model_path (str): YOLO 모델 경로
        """
        self.model = YOLO(model_path)
        self.grid_size = None
        self.image = None
        self.gray_image = None
        self.mask = None
        self.brightness_data = None
        self.brightness_matrix = None

    def calculate_grid_size(self, frame_height, frame_width):
        """
        프레임 해상도를 기반으로 grid size를 계산
        Args:
            frame_height (int): 프레임 높이
            frame_width (int): 프레임 너비
        Returns:
            int: 계산된 그리드 크기
        """
        return gcd(frame_height, frame_width)//10

    def process_frame(self, frame, rank):
        """
        프레임 처리 메인 함수
        Args:
            frame: 입력 프레임
        Returns:
            numpy.ndarray: 처리된 이미지
        """
        self.image = frame
        height, width = frame.shape[:2]
        
        # 그리드 크기 계산 (최초 1회)
        if self.grid_size is None:
            self.grid_size = self.calculate_grid_size(height, width)
            print(f"Calculated grid size: {self.grid_size}")

        # 1. 전체 이미지 그레이스케일 변환
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # 2. 전체 이미지 그리드 분석
        self.analyze_grid_brightness()
        
        # 3. YOLO 마스크 생성 (웨이퍼 영역 검출)
        self.detect_wafer_area()
        
        return self.create_single_rank_contour(rank)

    def analyze_grid_brightness(self):
        """
        전체 이미지의 그리드 밝기를 분석
        """
        height, width = self.gray_image.shape
        rows = height // self.grid_size
        cols = width // self.grid_size
        
        self.brightness_matrix = np.zeros((rows, cols))
        brightness_list = []
        
        for i in range(rows):
            for j in range(cols):
                y_start = i * self.grid_size
                y_end = min((i + 1) * self.grid_size, height)
                x_start = j * self.grid_size
                x_end = min((j + 1) * self.grid_size, width)
                
                cell = self.gray_image[y_start:y_end, x_start:x_end]
                avg_brightness = np.mean(cell)
                self.brightness_matrix[i, j] = avg_brightness
                brightness_list.append((avg_brightness, i, j))
        
        self.brightness_data = sorted(brightness_list, reverse=True)

    def detect_wafer_area(self):
        """
        YOLO 모델을 사용하여 웨이퍼 영역 검출
        """
        self.mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        results = self.model(self.image, task='segment', conf=0.8)
        
        detection_found = False
        for result in results:
            if result.masks is not None:
                for segment in result.masks.data:
                    detection_found = True
                    segment = segment.cpu().numpy()
                    segment = cv2.resize(segment, (self.image.shape[1], self.image.shape[0]))
                    segment = (segment > 0.5).astype(np.uint8) * 255
                    self.mask = cv2.bitwise_or(self.mask, segment)
        
        print("웨이퍼 탐지 {0}!".format("성공" if detection_found else "실패"))
        return detection_found

    def _create_brightness_ranges(self, min_brightness, max_brightness, num_groups=10):
        """
        명도 범위를 균등하게 나누는 메서드
        Args:
            min_brightness: 최소 명도값
            max_brightness: 최대 명도값
            num_groups: 생성할 그룹 수
        Returns:
            list: (시작값, 끝값) 튜플의 리스트
        """
        step = (max_brightness - min_brightness) / num_groups
        ranges = []
        for i in range(num_groups):
            # 밝은 부분부터 시작하도록 순서 변경
            start = max_brightness - ((i + 1) * step)
            end = max_brightness - (i * step)
            ranges.append((start, end))
            print(f"그룹 {i+1}: {start:.1f} ~ {end:.1f}")
        return ranges

    def _group_brightness_data(self, tolerance, min_pixel_ratio=0.00005):
        """
        밝기 데이터를 그룹화하고 픽셀 수가 적은 그룹 제거
        Args:
            tolerance: 밝기 값 허용 오차
            min_pixel_ratio: 전체 픽셀 대비 최소 필요 픽셀 비율 (0~1 사이 값)
        """
        # 전체 이미지 크기와 최소 필요 픽셀 수 계산
        total_pixels = self.image.shape[0] * self.image.shape[1]
        min_pixels = int(total_pixels * min_pixel_ratio)
        print(f"이미지 전체 픽셀 수: {total_pixels}")
        print(f"그룹당 최소 필요 픽셀 수: {min_pixels}")

        # 마스크 내의 유효한 픽셀들의 밝기값 수집
        valid_brightness = []
        for brightness, row, col in self.brightness_data:
            cell_mask = self.mask[row * self.grid_size:(row + 1) * self.grid_size,
                                col * self.grid_size:(col + 1) * self.grid_size]
            if np.any(cell_mask > 0):
                valid_brightness.append(brightness)
        
        if not valid_brightness:
            return []
        
        # 최소/최대 밝기값 찾기
        min_brightness = min(valid_brightness)
        max_brightness = max(valid_brightness)
        print(f"유효 픽셀 밝기 범위: {min_brightness:.1f} ~ {max_brightness:.1f}")
        
        # 밝기 범위 생성
        brightness_ranges = self._create_brightness_ranges(min_brightness, max_brightness)
        
        # 그룹화
        groups = [[] for _ in range(len(brightness_ranges))]
        
        # 각 셀을 해당하는 그룹에 할당
        for brightness, row, col in self.brightness_data:
            cell_mask = self.mask[row * self.grid_size:(row + 1) * self.grid_size,
                                col * self.grid_size:(col + 1) * self.grid_size]
            
            if not np.any(cell_mask > 0):
                continue
                
            # 해당 밝기값이 속하는 범위 찾기
            for i, (start, end) in enumerate(brightness_ranges):
                if start <= brightness <= end:
                    groups[i].append((brightness, row, col))
                    break
        
        # 픽셀 수가 적은 그룹 제거
        valid_groups = []
        for i, group in enumerate(groups):
            if len(group) >= min_pixels:
                valid_groups.append(group)
            else:
                print(f"그룹 {i+1} 제외됨 (픽셀 수: {len(group)})")
        
        print(f"최종 유효 그룹 수: {len(valid_groups)}")
        return valid_groups

    def create_single_rank_contour(self, target_rank, tolerance=0.1):
        """
        특정 순위의 밝기 영역만 표시
        Args:
            target_rank (int): 표시할 순위
            tolerance (float): 밝기 값 허용 오차
        Returns:
            numpy.ndarray: 결과 이미지
        """
        if not 1 <= target_rank <= 10:
            print("유효하지 않은 순위입니다. (1-10 사이 값을 입력하세요)")
            return self.image.copy()

        vis_img = self.image.copy()
        height, width = self.gray_image.shape
        
        # 순위별 그룹화
        brightness_groups = self._group_brightness_data(tolerance)
        if target_rank > len(brightness_groups):
            print(f"요청한 순위({target_rank})가 존재하는 그룹 수({len(brightness_groups)})를 초과합니다.")
            return vis_img
            
        # 해당 순위 그룹 처리
        target_group = brightness_groups[target_rank - 1]
        grid_mask = self._create_grid_mask(target_group, height, width)
        
        # 마스크 결합 및 컨투어 생성
        mask_final = cv2.bitwise_and(grid_mask, self.mask)
        return self._draw_contours(vis_img, mask_final, target_rank)

    def _create_grid_mask(self, group, height, width):
        """
        그리드 그룹에 대한 마스크 생성
        """
        grid_mask = np.zeros((height, width), dtype=np.uint8)
        for _, row, col in group:
            y_start = row * self.grid_size
            y_end = min((row + 1) * self.grid_size, height)
            x_start = col * self.grid_size
            x_end = min((col + 1) * self.grid_size, width)
            grid_mask[y_start:y_end, x_start:x_end] = 255
        return grid_mask

    def _draw_contours(self, image, mask, rank):
        """
        컨투어 그리기
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = self._define_colors()[rank - 1]
        
        cv2.drawContours(image, contours, -1, color, 3)
        
        # 큰 컨투어에만 라벨 추가
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(image, f"Rank {rank}", (cx-40, cy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)
        
        return image

    def _define_colors(self):
        """순위별 색상 정의"""
        return [
            (0, 0, 255),    # 빨강 (1등)
            (0, 165, 255),  # 주황 (2등)
            (0, 255, 255),  # 노랑 (3등)
            (0, 255, 0),    # 초록 (4등)
            (255, 0, 0),    # 파랑 (5등)
            (255, 0, 255),  # 마젠타 (6등)
            (128, 0, 0),    # 진한 파랑 (7등)
            (0, 128, 0),    # 진한 초록 (8등)
            (128, 128, 0),  # 올리브 (9등)
            (0, 0, 128),    # 진한 빨강 (10등)
        ]

    def visualize_grid(self):
        """
        그리드의 밝기 분포를 시각화
        Returns:
            numpy.ndarray: 시각화된 그리드 이미지
        """
        if self.brightness_matrix is None:
            print("Error: 그리드 분석이 수행되지 않았습니다.")
            return None

        height, width = self.gray_image.shape
        rows, cols = self.brightness_matrix.shape
        
        grid_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        cell_height = height // rows
        cell_width = width // cols
        
        valid_values = self.brightness_matrix[self.brightness_matrix > 0]
        if len(valid_values) > 0:
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            
            for i in range(rows):
                for j in range(cols):
                    y_start = i * cell_height
                    y_end = (i + 1) * cell_height
                    x_start = j * cell_width
                    x_end = (j + 1) * cell_width
                    
                    brightness = self.brightness_matrix[i, j]
                    if brightness > 0:
                        normalized_value = int(255 * (brightness - min_val) / (max_val - min_val))
                        
                        if normalized_value < 128:
                            b = 255 - 2 * normalized_value
                            g = 2 * normalized_value
                            r = 0
                        else:
                            b = 0
                            g = 255 - 2 * (normalized_value - 128)
                            r = 2 * (normalized_value - 128)
                        
                        grid_img[y_start:y_end, x_start:x_end] = [b, g, r]
                    
                    cv2.rectangle(grid_img, (x_start, y_start), (x_end, y_end), (128, 128, 128), 1)
        
        mask_overlay = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        grid_img = cv2.addWeighted(grid_img, 0.7 , mask_overlay, 0.3, 0)
        
        return grid_img

def main():
    processor = BrightnessProcessor("best.pt")
    video_path = "./wafer/warbling/warbling_0.1mm.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: 비디오 파일을 열 수 없습니다.")
        return
    
    ret, first_frame = cap.read()
    ret, first_frame = cap.read()
    ret, first_frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        return
            
    print("프레임 크기:", first_frame.shape)
    
    # grid_size 초기화 추가
    height, width = first_frame.shape[:2]
    processor.grid_size = processor.calculate_grid_size(height, width)
    print(f"Calculated grid size: {processor.grid_size}")
    
    # 프레임 처리
    processor.image = first_frame
    processor.gray_image = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    processor.analyze_grid_brightness()
    processor.detect_wafer_area()
    
    brightness_groups = processor._group_brightness_data(tolerance=0.1)
    valid_group_count = len(brightness_groups)
    
    print("\n===== 그룹 분석 결과 =====")
    print(f"실제 생성된 유효한 그룹 수: {valid_group_count}")
    
    if valid_group_count == 0:
        print("유효한 그룹이 없습니다.")
        return
    
    print(f"\n가능한 순위 범위: 1 ~ {valid_group_count}")
    
    while True:
        try:
            target_rank = int(input(f"\n표시할 순위를 입력하세요 (1-{valid_group_count}): "))
            if 1 <= target_rank <= valid_group_count:
                break
            else:
                print(f"잘못된 입력입니다. 1부터 {valid_group_count} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("숫자를 입력해주세요.")
            
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = 'wafer_analysis_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    ret, first_frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        return
            
    print("프레임 크기:", first_frame.shape)
    
    cv2.imshow('First Frame', first_frame)
    processor.process_frame(frame=first_frame, rank=target_rank)
    
    grid_viz = processor.visualize_grid()
    if grid_viz is not None:
        cv2.imshow('Grid Visualization', grid_viz)
    
    all_groups_vis = first_frame.copy()
    brightness_groups = processor._group_brightness_data(tolerance=0.1)
    colors = processor._define_colors()
    
    print(f"\n총 {len(brightness_groups)}개의 그룹이 발견되었습니다.")
    
    for rank, group in enumerate(brightness_groups):
        avg_brightness = np.mean([b for b, _, _ in group])
        print(f"Rank {rank + 1} 그룹의 평균 밝기: {avg_brightness:.1f}, 셀 개수: {len(group)}")
        
        height, width = processor.gray_image.shape
        grid_mask = np.zeros((height, width), dtype=np.uint8)
        
        for _, row, col in group:
            y_start = row * processor.grid_size
            y_end = min((row + 1) * processor.grid_size, height)
            x_start = col * processor.grid_size
            x_end = min((col + 1) * processor.grid_size, width)
            grid_mask[y_start:y_end, x_start:x_end] = 255
        
        mask_final = cv2.bitwise_and(grid_mask, processor.mask)
        
        kernel = np.ones((5,5), np.uint8)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[rank % len(colors)]
        cv2.drawContours(all_groups_vis, contours, -1, color, 3)
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"R{rank + 1}"
                    cv2.putText(all_groups_vis, label, (cx-20, cy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)
    
    cv2.imshow('All Brightness Groups', all_groups_vis)
    
    overlay = processor.create_single_rank_contour(target_rank)
    mask = cv2.cvtColor(cv2.absdiff(overlay, first_frame), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    color_overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        result = frame.copy()
        result[mask > 0] = color_overlay[mask > 0]
        
        cv2.imshow('Wafer Brightness Analysis', result)
        out.write(result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"비디오가 {output_path}로 저장되었습니다.")

if __name__ == "__main__":
    main()