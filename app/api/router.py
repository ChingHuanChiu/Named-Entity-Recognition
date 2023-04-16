from fastapi import APIRouter

from app.api.endpoints import ner

router = APIRouter()
router.include_router(ner.router)