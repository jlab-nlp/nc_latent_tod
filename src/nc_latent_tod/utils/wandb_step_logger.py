import logging
import pprint
from typing import Any, Dict, Optional

import wandb


def mark_run_complete(run: Optional[Any] = None):
    """
    Sets the `complete_run` tag on the given run, or the current one if called from a context that is already logging
    to `wandb` and passes no `run` argument
    """
    if not run:
        run = wandb.Api().run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
    run.tags.append("complete_run")
    run.update()


class WandbStepLogger:
    current_step: int
    log_for_step: Dict[str, Any]

    def __init__(self) -> None:
        super().__init__()
        self.current_step = 0
        self.log_for_step = {}

    def log(self, items: Dict[str, Any]) -> None:
        self.log_for_step.update(items)

    def step(self, increment: int = 1) -> None:
        if wandb.run is not None:
            wandb.log({
                "current_step": self.current_step,
                **self.log_for_step
            })
        else:
            if self.current_step == 0:
                logging.warning("WANDB NOT INITIALIZED! print-logging anything sent to WandbSteplLogger")
            pprint.pprint({
                "current_step": self.current_step,
                **self.log_for_step
            })
        self.current_step += increment
        self.log_for_step = {}
