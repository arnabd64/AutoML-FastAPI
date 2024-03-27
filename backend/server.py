from typing import Annotated, Any, Dict
import os
import time

from fastapi import Depends, FastAPI, status
from fastapi.requests import Request
from fastapi.responses import Response
from fastapi.background import BackgroundTasks
from fastapi.exceptions import HTTPException

from backend import utils
from backend.handlers import TokenHandler, StatusHandler


def startup_event():
    """
    Create a directory for storing artifacts needed
    by the Application
    """
    os.makedirs("./artifacts", exist_ok=True)


server = FastAPI(on_startup=[startup_event])


@server.get('/')
async def root():
    return {"message": "Welcome to the AutoML API"}


@server.get("/generate-token")
async def generate_token(token_handler: TokenHandler = Depends()):
    """
    __ENDPOINT__: `/generate-token`

    Generates a Job token for the user. The token is used
    to track the progress of the AutoML pipeline.

    ### Parameters
    - None

    ### Response
    - `token`: str - The generated token
    """
    return {"token": token_handler.generate()}


@server.post("/upload-dataset/{token}", dependencies=[Depends(utils.dataset_upload_pipeline)], status_code=status.HTTP_204_NO_CONTENT)
async def upload_dataset():
    pass


@server.post("/start-training/{token}", status_code=status.HTTP_204_NO_CONTENT)
async def start_training(background: BackgroundTasks,
                         training_args: Dict[str, Any] = Depends(utils.save_training_args)):
    background.add_task(utils.train_model, training_args)


@server.get("/check-status/{token}")
async def check_status(status_handler: Annotated[StatusHandler, Depends()]):
    content = status_handler.read_status()
    if content is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No status found", {"X-Token": status_handler.token})
    return content


@server.get("/evaluate-model/{token}")
async def evaluate_model(eval_dict: Annotated[Dict[str, float], Depends(utils.get_eval_results)]):
    return eval_dict


@server.get("/model-metadata/{token}")
async def model_metadata(token_handler: Annotated[TokenHandler, Depends()]):
    return utils.get_model_metadata(token_handler.token)


@server.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    response.headers["X-Process-Time"] = str(1000 * (time.perf_counter() - start_time))
    return response