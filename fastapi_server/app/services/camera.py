import cv2
import logging
from typing import List, Optional, Dict
from app.core.config import settings
import threading
from collections import deque
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, camera_index: int):
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            raise Exception(f"Failed to open camera {camera_index}")
            
        self.index = camera_index
        self.width = None
        self.height = None
        
        # 버퍼 설정 (thread-safe를 위해 deque 사용)
        self.BUFF_SIZE = 60
        self.buffer = deque(maxlen=self.BUFF_SIZE)
        
        self.test_resolutions = [
            (640, 480),   # VGA
            (800, 600),   # SVGA
            (1024, 768),  # XGA
            (1280, 720),  # HD
            (1280, 800),  # WXGA
            (1280, 1024), # SXGA
            (1920, 1080), # Full HD
            (2560, 1440), # QHD
            (3840, 2160)  # 4K UHD
        ]
        self.available_resolutions = []
        
        # 스레드 제어 플래그
        self.is_running = False
        self.thread = None
        
        # 카메라 해상도 설정
        
        # 버퍼링 시작
        # self.start_buffering()
        
    def start_buffering(self):
        """버퍼링 스레드 시작"""
        self.is_running = True
        self.thread = threading.Thread(target=self._buffer_frames)
        self.thread.daemon = True  # 메인 프로그램 종료시 스레드도 종료
        self.thread.start()
    
    def search_resolution(self):
        """카메라 해상도 조회"""
        self.available_resolutions = []
        
        for width, height in self.test_resolutions:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            ret, frame = self.camera.read()
            if ret and frame is not None:
                self.available_resolutions.append(f"{actual_width}x{actual_height}")
                print(f"{actual_width}x{actual_height} 사용 가능")
            else:
                print(f"{actual_width}x{actual_height} 사용 불가")
        
        self.available_resolutions = sorted(list(set(self.available_resolutions)), 
                                  key=lambda x: [int(i) for i in x.split('x')])
        
        return self.available_resolutions
    
    def set_resolution(self, width: int, height: int) -> bool:
        """카메라 해상도 설정"""
        try:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.width = width
            self.height = height
            return True
        except Exception as e:
            logger.error(f"Failed to set resolution: {e}")
            return False
    
    def _buffer_start(self):
        """버퍼링 시작"""
        self.start_buffering()
    
    def _buffer_frames(self):
        """프레임을 지속적으로 버퍼에 저장"""
        
        while self.is_running:
            success, frame = self.camera.read()
            if success:
                self.buffer.append(frame)
                
    def get_latest_frame(self) -> Optional[bytes]:
        """버퍼에서 최신 프레임 반환"""
        try:
            return self.buffer[-1] if self.buffer else None
        except IndexError:
            return None
            
    def get_frame_at_index(self, index: int) -> Optional[bytes]:
        """버퍼에서 특정 인덱스의 프레임 반환"""
        try:
            return self.buffer[index]
        except IndexError:
            return None
    
    def release(self):
        """카메라 및 스레드 정리"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.camera:
            self.camera.release()


class CameraManager:
    def __init__(self):
        self.camera_list: Dict[int, Camera] = {}
        self.connect_camera_list = []
        
        # 현재 사용중인 카메라
        self.current_index: Optional[int] = None
        self.current_camera: Optional[Camera] = None
        
    def find_camera(self) -> List[int]:
        """사용 가능한 카메라 인덱스 목록 반환"""
        for index in range(10):
            try:
                camera = cv2.VideoCapture(index)
                if camera.isOpened():
                    ret, frame = camera.read()
                    if ret and frame is not None:
                        self.connect_camera_list.append(index)
                        logger.info(f"Camera {index} is available")
                camera.release()
                time.sleep(0.05)
            except Exception as e:
                logger.warning(f"Error checking camera {index}: {e}")
                
        return self.connect_camera_list
    
    def set_camera(self, index):
        self.current_index = index
        self.current_camera = self.camera_list.get(index)
        
    def get_camera(self, index: int) -> Optional[Camera]:
        if camera := self.camera_list.get(index):
            return camera
        return None

    def add_camera(self, index: int) -> Camera:
        # 이미 존재하는 카메라인지 확인
        if existing_camera := self.camera_list.get(index):
            print(f"Camera {index} already exists")
            return existing_camera
        
        time.sleep(0.1)  # 100ms 대기
        # 새 카메라 생성 및 저장
        self.camera_list[index] = Camera(index)
        return self.camera_list[index]
        
    def get_using_camera(self, index: int):
        return self.current_camera


            
    def current_release_camera(self, index: int):
        if camera := self.camera_list.get(index):
            camera.release()
            del self.camera_list[index]
            if self.current_index == index:
                self.current_index = None
                
    def all_release_camera(self):
        for camera in self.camera_list.values():
            camera.release()
        self.camera_list.clear()
        self.connect_camera_list = []
        
        self.current_index: Optional[int] = None
        self.current_camera: Optional[Camera] = None

