import uvicorn
from api.v1.api_route import router
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import logging

from app.app import load_models
from transformers import VectorizeTransformer, TfidfVectorizeTransformer


load_models()

app = FastAPI(
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )


app.include_router(router, prefix="/api/v1/models")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
