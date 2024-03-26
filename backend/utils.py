import json
import os
from typing import Annotated, Any, Dict
from io import BytesIO

import pandas as pd
from fastapi import Form, UploadFile, status
from fastapi.exceptions import HTTPException

from backend import automl
from backend.status_handler import StatusHandler
from backend.models import ArtifactPaths, Task


def startup_event():
    """
    Create a directory for storing artifacts needed
    by the Application
    """
    os.makedirs("./artifacts", exist_ok=True)
    

def dataset_upload_pipeline(token: str, csv: UploadFile):
    # verify if uploaded file is CSV
    if not csv.filename.endswith('.csv'):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Only CSV files are allowed",
                            {"X-Token": token})
    
    df = pd.read_csv(BytesIO(csv.file.read()))
    df = automl.dataset_preprocess(df)
    df.to_parquet(ArtifactPaths.DATASET.value.format(token=token), index=False)
    StatusHandler(token).save_status("Dataset uploaded successfully")


def save_training_args(token: str, 
                       target: Annotated[str, Form(...)], 
                       task: Annotated[Task, Form(...)],
                       iterations: Annotated[int, Form(...)]):
    training_args = {
        "token": token,
        "target": target,
        "task": task.value,
        "iterations": iterations
    }

    with open(ArtifactPaths.TRAINING_ARGS.value.format(token=token), "w") as args:
        json.dump(training_args, args)

    StatusHandler(token).save_status("Training arguments saved successfully")
    return training_args


def train_model(training_args: Dict[str, Any]):
    token = training_args["token"]

    status_handler = StatusHandler(token)

    df = pd.read_parquet(ArtifactPaths.DATASET.value.format(token=token))
    status_handler.save_status("Dataset imported in Parquet format")

    model = automl.trainer(df, training_args)
    status_handler.save_status("Model trained successfully")

    eval_results = automl.evaluate_model(model, df, training_args)
    with open(ArtifactPaths.EVALUATION.value.format(token=token), "w") as eval_file:
        json.dump(eval_results, eval_file)
    status_handler.save_status("Evaluation done successfully")

    model.pickle(ArtifactPaths.MODEL.value.format(token=token))
    status_handler.save_status("Model saved successfully")


def get_eval_results(token: str) -> Dict[str, float]:
    filepath = ArtifactPaths.EVALUATION.value.format(token=token)
    if not os.path.exists(filepath):
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            "Evaluation not yet done",
                            {"X-Token": token})
    
    with open(ArtifactPaths.EVALUATION.value.format(token=token)) as eval_file:
        return json.load(eval_file)