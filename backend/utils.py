import os
import json

import pandas as pd
from fastapi import UploadFile, status, Form
from fastapi.exceptions import HTTPException

from backend import automl
from backend.models import ArtifactPaths, Task, Iterations
from typing import Annotated, Dict, Any


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
    
    # load the CSV into dataframe
    df = pd.read_csv(csv.file.read())

    # preprocess the data
    df = automl.dataset_preprocess(df)

    # export the dataframe to parquet
    df.to_parquet(ArtifactPaths.DATASET.value.format(token=token), index=False)


def save_training_args(token: str, 
                       target: Annotated[str, Form(...)], 
                       task: Annotated[Task, Form(...)],
                       iterations: Annotated[Iterations, Form(...)]):
    training_args = {
        "token": token,
        "target": target,
        "task": task.value,
        "iterations": iterations.value
    }

    with open(ArtifactPaths.TRAINING_ARGS.value.format(token=token), "w") as args:
        json.dump(training_args, args)

    return training_args


def train_model(training_args: Dict[str, Any]):
    # get the token
    token = training_args["token"]

    # load the dataset
    df = pd.read_parquet(ArtifactPaths.DATASET.value.format(token=token))

    # train the model
    model = automl.trainer(df, training_args)

    # do evaluations
    eval_results = automl.evaluate_model(model, df, training_args)

    # save the evaluation results
    with open(ArtifactPaths.EVALUATION.value.format(token=token), "w") as eval_file:
        json.dump(eval_results, eval_file)

    # save the model
    model.pickle(ArtifactPaths.MODEL.value.format(token=token))