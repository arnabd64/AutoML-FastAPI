import json
import os
import secrets
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import UploadFile, status, Depends
from fastapi.exceptions import HTTPException

from backend.automl import preprocess, export_dataset
from backend.models import ArtifactPaths


class TokenHandler:

    __slots__ = ("token")

    def __init__(self, token: Optional[str] = None):
        self.token = token


    def generate(self):
        return secrets.token_hex(nbytes=8)
    

class StatusHandler:

    __slots__ = ("token", "filepath")

    def __init__(self, token_handler: TokenHandler = Depends()):
        self.token = token_handler.token
        self.filepath = ArtifactPaths.STATUS.value.format(token=self.token)


    @property
    def timestamp(self):
        return str(datetime.now())
    

    @property
    def status_exists(self):
        return os.path.exists(self.filepath)
    

    def read_json(self) -> List[Dict[str, Any]]:
        with open(self.filepath, "r") as fp:
            return json.load(fp)
        

    def write_json(self, data: List[Dict[str, Any]]):
        with open(self.filepath, "w") as fp:
            json.dump(data, fp)
        

    def read_status(self):
        return self.read_json() if self.status_exists else None
        

    def save_status(self, message: str, flag: str = "OK", extras: Optional[Dict[str, Any]] = None):
        status_history = self.read_status() or list()
        status = {"flag": flag, "message": message, "time": self.timestamp}

        if extras:
            status.update(extras)

        status_history.append(status)
        self.write_json(status_history)
        return status


class UploadHandler:

    __slots__ = ("token", "csv")

    def __init__(self, csv: UploadFile, token_handler: TokenHandler = Depends()):
        # validations
        if csv.content_type != "text/csv":
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid file type")
        
        self.token = token_handler.token
        self.df = pd.read_csv(BytesIO(csv.file.read()))


    def __call__(self):
        # preprocess the dataframe
        self.df = preprocess(self.df)

        # export the dataframe
        export_dataset(self.df, ArtifactPaths.DATASET.value.format(token=self.token))

        return StatusHandler(self.token).save_status("Dataset uploaded successfully")
    