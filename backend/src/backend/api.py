from fastapi import APIRouter

router = APIRouter()


@router.get("/health", response_model=dict)
def read_root():
    return {"status": "ok"}
