from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from backend import utils
from typing import Dict, Any
import secrets


server = FastAPI(on_startup=[utils.startup_event],
                 default_response_class=PlainTextResponse)


@server.get('/')
async def root():
    return "server is running"


@server.get("/generate-token")
async def generate_token():
    return secrets.token_hex(nbytes=8)


@server.post("/upload-dataset/{token}", dependencies=[Depends(utils.dataset_upload_pipeline)])
async def upload_dataset():
    return "Upload successful"


@server.post("/start-training/{token}")
async def start_training(background: BackgroundTasks,
                         training_args: Dict[str, Any] = Depends(utils.save_training_args)):
    background.add_task(utils.train_model, training_args)
    return "started training"


@server.get("/check-status/{token}", response_class=JSONResponse)
async def check_status(token: str):
    return {"status": "completed"}


@server.get("/evaluate-model/{token}", response_class=JSONResponse)
async def evaluate_model(token: str):
    return {"accuracy": 0.95}