from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # 프로젝트 기본 정보
    PROJECT_NAME: str = "Camera Detection API"
    PROJECT_DESCRIPTION: str = ""
    VERSION:str = "1.0"
    
    # API 설정
    API_STR: str = "/api"
    
    # CORS 설정
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8877",
        "http://127.0.0.1:8877",
        "http://100.80.66.12:8877",
    ]
    
    # 파일 경로 설정
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TEMP_DIR: str = os.path.join(BASE_DIR, "temp")
    IMG_DIR: str = os.path.join(TEMP_DIR, "img")
    INPUT_DIR: str = os.path.join(TEMP_DIR, "input")
    OUTPUT_DIR: str = os.path.join(TEMP_DIR, "output")
    
    # 이미지 관련 설정
    IMAGE_INPUT_DIR: str = os.path.join(INPUT_DIR, "image")
    IMAGE_OUTPUT_DIR: str = os.path.join(OUTPUT_DIR, "image")
    
    # 비디오 관련 설정
    VIDEO_INPUT_DIR: str = os.path.join(INPUT_DIR, "video")
    VIDEO_OUTPUT_DIR: str = os.path.join(OUTPUT_DIR, "video")
    DETECTION_VIDEO_DIR: str = os.path.join(VIDEO_OUTPUT_DIR, "detection_video")
    FULL_VIDEO_DIR: str = os.path.join(VIDEO_OUTPUT_DIR, "full_video")
    
    # 카메라 설정
    FRAME_RESIZE_DIMENSIONS: tuple = (800, 800)
    VIDEO_FPS: int = 60
    
    class Config:
        case_sensitive = True

# 전역 설정 객체 생성
settings = Settings()

# 필요한 디렉토리 생성
for directory in [
    settings.TEMP_DIR,
    settings.INPUT_DIR,
    settings.OUTPUT_DIR,
    settings.IMAGE_INPUT_DIR,
    settings.IMAGE_OUTPUT_DIR,
    settings.VIDEO_INPUT_DIR,
    settings.VIDEO_OUTPUT_DIR,
    settings.DETECTION_VIDEO_DIR,
    settings.FULL_VIDEO_DIR,
]:
    os.makedirs(directory, exist_ok=True)