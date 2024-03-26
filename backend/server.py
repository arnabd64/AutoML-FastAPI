import secrets
from typing import Annotated, Any, Dict

from fastapi import Depends, FastAPI, status
from fastapi.background import BackgroundTasks
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from backend import utils
from backend.status_handler import StatusHandler

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
async def check_status(status_history: Annotated[StatusHandler, Depends()]):
    content = status_history.read_status()
    if content is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No status found", {"X-Token": status_history.token})
    return content


@server.get("/evaluate-model/{token}", response_class=JSONResponse)
async def evaluate_model(eval_dict: Annotated[Dict[str, float], Depends(utils.get_eval_results)]):
    return eval_dict
