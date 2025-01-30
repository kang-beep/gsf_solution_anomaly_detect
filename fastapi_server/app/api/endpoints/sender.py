import io
import os
import cv2
import asyncio
import requests
import numpy as np
import asyncio
from PIL import Image
from fastapi import APIRouter

from app.services import camera_service
from app.core.config import settings

router = APIRouter()

# FastAPI 엔드포인트에서
@router.get("/start_sending")
async def start_image_sending():
    if not camera_service.cam_save_flag:
        camera_service.cam_save_flag = True
        asyncio.create_task(send_images())
        return {"status": "started",
                "message" : "Recording started"
                }
    return {"status": "already running"}

@router.get("/stop_sending")
async def stop_image_sending():
    camera_service.cam_save_flag = False
    return {"status": "stopped"}

async def send_images():
    """생성된 이미지를 anomaly server로 전송"""
    PORT = '8877'
    SERVER_URL = f'http://127.0.0.1:{PORT}/monitor/'
    resize_dim = (800, 800)
    frame_count = 0

    print(f'이미지 전송 시작 -> AD_SERVER ({SERVER_URL})')

    try:
        while camera_service.cam_save_flag:
            # ensure_camera_open을 사용하여 카메라 상태 확인 및 재초기화
            camera = camera_service.ensure_camera_open(camera_service.current_using_index)
            if not camera:
                return {"error": "Failed to initialize camera"}
            
            frame = camera.get_frame()
            if frame is not None:
                try:
                    # bytes를 numpy array로 변환
                    nparr = np.frombuffer(frame, np.uint8)
                    opencv_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
                    # OpenCV 프레임을 PIL Image로 변환
                    pil_image = Image.fromarray(cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB))
                    pil_image = pil_image.resize(resize_dim, Image.LANCZOS)
                    
                    # 이미지를 바이트로 변환
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    
                    # 전송할 파일 데이터 준비
                    filename = f'{frame_count:03d}.png'
                    files = {
                        'image': (filename, img_byte_arr.getvalue(), 'image/png')
                    }
                    
                    # 서버로 전송
                    response = requests.post(SERVER_URL, files=files)
                    if response.status_code == 200:
                        print(f"Image {frame_count:03d} sent successfully")
                        if os.path.exists(filename):
                            os.remove(filename)
                    else:
                        print(f"Failed to send image {frame_count}")
                    
                    frame_count += 1

                except Exception as e:
                    print(f"Error processing/sending frame {frame_count}: {e}")

            await asyncio.sleep(0.1)  # 10 FPS로 제한

    except Exception as e:
        print(f"Error in send loop: {e}")
    finally:
        print("Image sending stopped")
        camera_service.cam_save_flag = False