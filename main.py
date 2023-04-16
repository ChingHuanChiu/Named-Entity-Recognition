import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


from app.api.router import router as api_router


tags_metadata = [
    {
        "name": "NER",
        "description": "NER",
    },
]


app = FastAPI(openapi_tags=tags_metadata,
              version='0.0.0',
              title="NER"
              )


origins = ["http://localhost"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]

)

app.include_router(api_router)
