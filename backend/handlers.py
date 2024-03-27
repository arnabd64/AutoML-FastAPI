import json
import os
import secrets
from datetime import datetime
from typing import Dict, List, Optional, Any

from backend.models import ArtifactPaths


class TokenHandler:

    __slots__ = ("token")

    def __init__(self, token: Optional[str] = None):
        self.token = token


    def generate(self):
        self.token = secrets.token_hex(nbytes=8)
        return self.token
    

    def validate(self):
        pass


class StatusHandler:

    __slots__ = ("token", "filepath")

    def __init__(self, token: str):
        self.token = token
        self.filepath = ArtifactPaths.STATUS.value.format(token=token)


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
        

    def save_status(self, message: str, flag: Optional[str] = None, extras: Optional[Dict[str, Any]] = None):
        status_history = self.read_status() or list()
        status = {"message": message, "time": self.timestamp}
        if flag:
            status["flag"] = flag

        if extras:
            status.update(extras)

        status_history.append(status)
        self.write_json(status_history)

