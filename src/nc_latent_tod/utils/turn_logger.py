import abc
import logging
from abc import ABC
from typing import List, Any

import wandb

from nc_latent_tod.data_types import DatasetTurnLog
from nc_latent_tod.utils.general import write_json


class AbstractLogViewer(metaclass=abc.ABCMeta):
    """
    An abstract class for somehow viewing or interpretting logged turns. For example, one could read only the DST
    relevant portions and log them to a txt file in a human readable format, or do the same but to a CSV for viewing
    in excel, etc.
    """

    @abc.abstractmethod
    def view_turn(self, log: DatasetTurnLog, **kwargs) -> Any:
        """
        Do something with the given turn log to make it 'viewable' in some way. For example, a text and log viewer will
        print useful items from the turn relavant to a task, and log those to a file.
        """
        pass

    def view_turns(self, logs: List[DatasetTurnLog], **kwargs) -> Any:
        # implementers may parallelize. They'll have to deal with kwargs themselves
        for log in logs:
            self.view_turn(log, **kwargs)


class WandbArtifactViewerMixin(ABC):

    @abc.abstractmethod
    def to_wandb_artifact(self) -> wandb.Artifact:
        pass


class TextFileLogViewer(AbstractLogViewer, WandbArtifactViewerMixin, ABC):

    logger: logging.Logger
    task: str
    log_file_path: str
    file_handler: logging.FileHandler
    step: int = 0

    def __init__(self, log_file_path: str, task: str, step: int = 0, log_level: int = logging.DEBUG,
                 log_file_mode: str = 'w', add_stream_handler: bool = False) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.propagate = False  # this is an atypical logger: it just helps us format a file
        self.task = task or '<no_task_specified>'
        self.log_file_path = log_file_path
        self.logger.setLevel(log_level)
        formatter: logging.Formatter = logging.Formatter(
            '%(task)s - %(dialogue_id)s - %(turn_id)s - %(message)s')
        self.file_handler = logging.FileHandler(log_file_path, mode=log_file_mode)
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)
        if add_stream_handler:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
        self.step = step

    def log_with_turn_metadata(self, log_string: str, log: DatasetTurnLog) -> None:
        self.logger.info(log_string, extra={
            'task': self.task, 'dialogue_id': log['dialogue_id'], 'turn_id': log['turn_id']
        })

    def to_wandb_artifact(self) -> wandb.Artifact:
        import wandb  # local import to allow use outside of wandb runs
        self.file_handler.flush()  # make sure all logs are written to file
        artifact_name: str = f"{wandb.run.name}_{self.task}_logs"
        if self.step:
            artifact_name += f"_{self.step}"
        artifact: wandb.Artifact = wandb.Artifact(artifact_name, type=f"{self.task}_log_file")
        artifact.add_file(self.log_file_path)
        return artifact


class TurnLogger:

    running_log: List[DatasetTurnLog]
    viewers: List[AbstractLogViewer]

    def __init__(self, running_log: List[DatasetTurnLog] = None,
                 viewers: List[AbstractLogViewer] = None) -> None:
        self.running_log = running_log or []
        self.viewers = viewers or []

    def log_turn(self, turn: DatasetTurnLog, **kwargs) -> None:
        self.running_log.append(turn)
        for viewer in self.viewers:
            viewer.view_turn(turn, **kwargs)

    # take an unspecified number of viewer arguments
    def add_viewers(self, viewers: List[AbstractLogViewer]) -> None:
        if viewers:
            self.viewers.extend(viewers)

    def write_running_log(self, path: str) -> None:
        write_json(self.running_log, path)
