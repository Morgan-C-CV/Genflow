from fastapi import APIRouter
from app.api.v1.endpoints import agent, runtime, search

api_router = APIRouter()
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(runtime.router, prefix="/runtime", tags=["runtime"])
