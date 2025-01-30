from fastapi import APIRouter
import logging

from app.services import camera_service
from pydantic import BaseModel
from fastapi import Form

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/available")
async def get_available_cameras() -> list:
    """사용 가능한 카메라 목록 조회"""
    # 등록된 카메라 모두 메모리 해제
    camera_service.all_release_camera()
    
    # 연결된 카메라 목록 조회
    available_cameras = camera_service.find_camera()
    
    return available_cameras

@router.post("/add_camera")
async def add_camera(camera_index: int) -> dict:
    """카메라 추가"""
    camera = camera_service.add_camera(camera_index)
    if camera:
        # 카메라 해상도 반환
        resolutions = camera.search_resolution()
        return {
            "status": "success",
            "resolutions": resolutions
        }
    return {
        "status": "error",
        "message": f"Camera {camera_index} is not available"
    }

@router.post("/open_camera")
async def open_camera(
    camera_index: int = Form(...), 
    width: int = Form(...), 
    height: int = Form(...)) -> dict:
    """카메라 열기"""
    
    # 사용 카메라 설정
    
    camera = camera_service.get_camera(camera_index)
    camera_service.set_camera(camera_index)
    camera.set_resolution(width, height)
    camera.start_buffering()
    
    success = True
    
    if camera:
        if success:
            print("카메라 열기 성공")
            return {
                "status": "success",
                "message": f"Camera {camera_index} opened with resolution {width}x{height}"
            }
            
        else:
            print("카메라 열기 실패")
            return {
            "status": "error",
            "message": f"Failed to open camera {camera_index} with resolution {width}x{height}"
        }
    print("카메라 없음")
    return {
        "status": "error",
        "message": f"Camera {camera_index} not found"
    }


@router.post("/release/all")
async def release_all():
    """모든 카메라 해제"""
    camera_service.release_all_cameras()
    return {"message": "All cameras released"}

