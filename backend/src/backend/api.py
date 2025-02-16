from fastapi import APIRouter

router = APIRouter()


@router.get("/", response_model=str)
def read_root():
    return "API is running!"
