from fastapi import APIRouter

router = APIRouter()

# 서버 연결상태 확인 
@router.get("/health_check")
async def health_check():
    return {"status" : "ok"}