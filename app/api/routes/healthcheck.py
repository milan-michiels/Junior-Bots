from fastapi import APIRouter

health_router = APIRouter()


@health_router.get("/")
def healthcheck():
    return {"status": "ok"}
