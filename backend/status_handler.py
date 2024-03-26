import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from backend.models import ArtifactPaths


class StatusHandler:

    __slots__ = ("token", "filepath")

    def __init__(self, token: str):
        self.token = token
        self.filepath = ArtifactPaths.STATUS.value.format(token=token)


    @property
    def timestamp(self):
        return str(datetime.now())
    

    def read_status(self) -> Optional[List[Dict[str, str]]]:
        if not os.path.exists(self.filepath):
            return None
        with open(self.filepath, "r") as status:
            return json.load(status)
        

    def save_status(self, message: str):
        # read prior status
        status_history = self.read_status()
        if status_history is None:
            status_history = list()

        # add the current status
        status_history.append({"message": message, "time": self.timestamp})
        
        # write the status to file
        with open(self.filepath, "w") as status:
            json.dump(status_history, status)

