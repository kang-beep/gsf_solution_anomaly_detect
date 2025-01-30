from .camera import CameraManager
from .image import ImageProcessor, BatchProcessor
# from .video import VideoProcessor

# 싱글톤 인스턴스 생성
camera_service = CameraManager()
image_processor = ImageProcessor()
batch_processor = BatchProcessor()