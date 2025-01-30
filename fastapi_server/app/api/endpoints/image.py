import os
import glob
import cv2
import numpy as np
from typing import List
from fastapi import APIRouter, Body, Query, Response
from fastapi.responses import StreamingResponse
import io
import logging

from app.services import camera_service
from app.services import batch_processor
from app.schemas.models import DataItem
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/image_feed")
async def get_image_feed(camera_index: int = Query(description="카메라 인덱스")):
    """단일 이미지 캡처"""
    # 저장 경로 설정
    save_dir = settings.IMG_DIR
    save_path = os.path.join(save_dir, "stop.png")
    
    # 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 카메라 스트림 가져오기
    camera_stream = camera_service.get_camera(camera_index)
    if not camera_stream:
        return {"error": "Camera not found"}

    # 프레임 읽기 시도
    max_attempts = 50
    for attempt in range(max_attempts):
        frame = camera_stream.get_frame()
        if frame is not None:
            with open(save_path, 'wb') as f:
                f.write(frame)
            logger.info(f"Request : Successfully saved image to {save_path}")
            
            # StreamingResponse 대신 일반 Response 사용
            return Response(
                content=frame,
                media_type="image/png",
                headers={
                    "Content-Disposition": "inline; filename=image.png"
                }
            )

    return {"error": "Failed to capture image"}


@router.post("/rgb2hsv")
async def convert_rgb_to_hsv(rgb: dict = Body(...)):
    """RGB 색상을 HSV 색상으로 변환"""
    try:
        r, g, b = rgb['r'], rgb['g'], rgb['b']
        rgb_array = np.uint8([[[b, g, r]]])  # BGR 순서로 변환
        hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_array[0][0]
        
        return {
            "h": int(h),  # 0-179
            "s": int(s),  # 0-255
            "v": int(v)   # 0-255
        }
    except KeyError:
        return {"error": "Invalid RGB values. Required keys: 'r', 'g', 'b'"}
    except Exception as e:
        return {"error": f"Conversion failed: {str(e)}"}

@router.post("/process")
async def process_image(table_data: List[DataItem] = Body(...)):
    """테이블 데이터 기반 이미지 처리"""
    try:
        print("이미지 저장")
        # 임시 저장된 이미지 경로
        source_image = os.path.join(settings.IMG_DIR, "stop.png")
        if not os.path.exists(source_image):
            logger.error(f"Error path: {str(e)}")
            return {"error": "No source image found"}

        # 기존 결과 파일 정리(삭제)
        output_dir = os.path.join(settings.OUTPUT_DIR, "image")
        os.makedirs(output_dir, exist_ok=True)
        
        for file in glob.glob(os.path.join(output_dir, "*.png")):
            try:
                os.remove(file)
            except Exception as e:
                logger.error(f"Error deleting {file}: {str(e)}")

        print("데이터 형식 변환")
        # 데이터 형식 변환
        processed_data = [{
            "lower_bound": {"h": item.lower_bound.h, "s": item.lower_bound.s, "v": item.lower_bound.v},
            "upper_bound": {"h": item.upper_bound.h, "s": item.upper_bound.s, "v": item.upper_bound.v},
            "margin": item.margin
        } for item in table_data]

        print("일괄 처리 실행")
        # 일괄 처리 실행
        results = batch_processor.process_image_batch(
            image_path=source_image,
            table_data=processed_data,
        )

        process_image_path = []
        
        print("이미지 파일 생성")
        # 이미지 파일 생성
        for filename, img_data in results["image_data"]:
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'wb') as f:
                f.write(img_data)
            
            if "crop_image" in filename:
                relative_path = os.path.join("/temp/output/image", filename)
                process_image_path.append(relative_path)
                
        print("결과 반환", process_image_path)
        # 결과 반환
        return {
            "status": "success" if not results["failures"] else "partial_success",
            "processed_rows": len(table_data),
            "successful_detections": len(results["processed_images"]),
            "failed_detections": results["failures"],
            "alert_message": "HSV값을 조절해주세요" if results["failures"] else None,
            "image_paths": process_image_path,
        }

    except Exception as e:
        print("에러")
        logger.error(f"Image processing failed: {str(e)}")
        return {"error": f"Image processing failed: {str(e)}"}
    
    
@router.get("/save/start")
async def start_image_saving():
    """이미지 저장 시작"""
    try:
        if not camera_service.cam_save_flag:
            camera_service.cam_save_flag = True
            
            # 저장 디렉토리 설정
            output_dir = os.path.join(settings.OUTPUT_DIR, "image/saving_img")
            os.makedirs(output_dir, exist_ok=True)

            # 기존 파일 정리
            for file in glob.glob(os.path.join(output_dir, "*.png")):
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error deleting {file}: {str(e)}")

            return {"status": "started", "save_dir": output_dir}
        return {"status": "already_running"}

    except Exception as e:
        return {"error": f"Failed to start image saving: {str(e)}"}

@router.get("/save/stop")
async def stop_image_saving():
    """이미지 저장 중지"""
    camera_service.cam_save_flag = False
    return {"status": "stopped"}

@router.get("/latest")
async def get_latest_image(camera_index: int):
    """현재 카메라의 최신 이미지 가져오기"""
    try:
        camera = camera_service.get_camera(camera_index)
        if not camera:
            return {"error": "Camera not found"}

        frame = camera.get_latest_frame()
        if frame is None:
            return {"error": "Failed to capture image"}

        return StreamingResponse(io.BytesIO(frame), media_type="image/png")

    except Exception as e:
        return {"error": f"Failed to get latest image: {str(e)}"}

@router.get("/current")
async def get_current_status():
    """현재 이미지 처리 상태 조회"""
    return {
        "save_flag": camera_service.cam_save_flag,
        "active_camera": camera_service.current_using_index
    }