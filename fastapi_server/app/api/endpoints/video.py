import os
import cv2
import asyncio
import datetime
from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.services import camera_service

import logging
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/feed")
async def get_video_feed(
    request: Request,  # 1. Request 매개변수 추가
    camera_index: int = Query(description="카메라 인덱스")
    ):

    camera = camera_service.get_camera(camera_index)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    return StreamingResponse(
        generate_frames(request, camera),  # 3. request 매개변수 추가
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
    

async def generate_frames(request: Request, camera):  # 4. request 매개변수 추가
    try:
        while True:
            # 5. 클라이언트 연결 상태 확인 로직 추가
            if await request.is_disconnected():
                logger.info("Client disconnected, stopping frame generation")
                break

            frame = camera.get_latest_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
                
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                await asyncio.sleep(0.01)
                continue
                
            # 6. 연결 오류 처리 추가
            try:
                yield (b'--frame\r\n'
                       b'Content-Type image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except (ConnectionResetError, RuntimeError):
                logger.info("Connection reset by client")
                break
    except Exception as e:
        logger.error(f"Error in generate_frames: {str(e)}", exc_info=True)
            
            
@router.get("/stop_feed")
async def stop_feed():
    """비디오 스트림 일시 중지 중지"""
    camera = camera_service.get_camera(camera_service.current_index)
    frame = camera.get_latest_frame()
    cv2.imwrite(f"{settings.TEMP_DIR}/stop.jpg", frame)
    
    return {"status": "stopped"}
            
# @router.get("/full/start")
# async def start_full_video_recording():
#     """전체 영상 녹화 시작"""
#     try:
#         logger.info("Starting video recording...")
        
#         if camera_manager.cam_save_flag:
#             logger.warning("Recording already in progress")
#             return {"error": "Recording already in progress"}

#         logger.info(f"Getting camera with index: {camera_manager.current_using_index}")
#         # ensure_camera_open을 사용하여 카메라 상태 확인 및 재초기화
#         camera = camera_manager.ensure_camera_open(camera_manager.current_using_index)
#         if not camera:
#             logger.error("Failed to initialize camera")
#             return {"error": "Failed to initialize camera"}

#         logger.info("Setting up output path...")
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_dir = os.path.join(settings.VIDEO_OUTPUT_DIR, "full_video")
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, f'video_{timestamp}.mp4')
#         logger.info(f"Output path: {output_path}")

#         logger.info("Creating background task...")
#         asyncio.create_task(camera_manager.record_video(camera, output_path))

#         return {
#             "status": "success",
#             "message": "Recording started",
#             "output_path": output_path,
#             "fps": settings.VIDEO_FPS
#         }

#     except Exception as e:
#         logger.error(f"Error in start_full_video_recording: {str(e)}", exc_info=True)
#         camera_manager.cam_save_flag = False
#         raise HTTPException(status_code=500, detail=str(e))


# @router.get("/detection/start")
# async def start_detection_video_recording():
#     """검출 영역 녹화 시작"""
#     try:
#         logger.info(f"Starting detection recording with camera index: {camera_manager.current_using_index}")
        
#         if camera_manager.cam_save_flag:
#             return {"error": "Recording already in progress"}
        
#         # ensure_camera_open을 사용하여 카메라 상태 확인 및 재초기화
#         camera = camera_manager.ensure_camera_open(camera_manager.current_using_index)
#         if not camera:
#             logger.error("Failed to initialize camera")
#             return {"error": "Failed to initialize camera"}

#         if not hasattr(batch_processor, 'contours_list') or not batch_processor.contours_list :
#             raise HTTPException(status_code=400, detail="No detection area configured")
        
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_dir = os.path.join(settings.VIDEO_OUTPUT_DIR, "detection_video")
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, f'detection_video_{timestamp}.mp4')
        
#         asyncio.create_task(camera_manager.record_detection_video(
#             camera,
#             output_path,
#             batch_processor,
#             image_processor,
#             settings.VIDEO_FPS,
#         ))
        
#         return {
#             "status" : "success",
#             "message" : "Recording started",
#             "output_path": output_path,
#             "fps": settings.VIDEO_FPS
#         }
    
#     except Exception as e:
#         camera_manager.cam_save_flag = False
#         logger.error(f"Detection recording error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))




# @router.get("/status")
# async def get_recording_status():
#     """현재 녹화 상태 확인"""
#     return {
#         "is_recording": camera_manager.cam_save_flag,
#         "current_camera": camera_manager.current_using_index
#     }