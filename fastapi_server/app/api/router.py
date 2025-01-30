from fastapi import APIRouter
from .endpoints import camera, image, video, sender, health

api_router = APIRouter()

# 각 엔드포인트의 라우터
api_router.include_router(
    camera.router,
    prefix="/camera",
    tags=["camera"],
)

api_router.include_router(
    image.router,
    prefix="/image",
    tags=["image"],
)

api_router.include_router(
    video.router,
    prefix="/video",
    tags=["video"],
)

api_router.include_router(
    sender.router,
    prefix="/sender",
    tags=["sender"],
)

api_router.include_router(health.router, prefix="/health", tags=["health"])