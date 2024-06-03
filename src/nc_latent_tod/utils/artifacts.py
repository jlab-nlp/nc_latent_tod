import json
import os
from typing import Any, Optional, List

import pandas as pd
import wandb

from nc_latent_tod.data_types import DatasetTurnLog
from nc_latent_tod.utils.general import NC_LATENT_TOD_OUTPUTS_DIR


def output_dir_to_run_or_artifact_name(output_dir: str) -> str:
    parent_dir = NC_LATENT_TOD_OUTPUTS_DIR in os.environ and os.environ[NC_LATENT_TOD_OUTPUTS_DIR] or "outputs/"
    run_name: str = output_dir.replace("../expts/", "").replace(parent_dir, "").replace('/', '-')\
        .replace("-data-users-bking2-rec_dst-expts-runs", "")
    run_name = run_name.replace(".json", "")
    run_name = run_name.replace("runs-", "",1)
    if run_name.startswith("-"):
        return run_name[1:]
    return run_name


def get_json_artifact_by_file_name(expected_file_path: str) -> Any:
    output_dir, file_name = expected_file_path.rsplit("/", maxsplit=1)
    artifact_name: str = output_dir_to_run_or_artifact_name(output_dir)
    try:
        return read_json_artifact(artifact_name, file_name)
    except BaseException as e:
        artifact_name = "-data-users-bking2-rec_dst-expts-runs" + artifact_name
        return read_json_artifact(artifact_name, file_name)


def read_json_artifact(artifact_name: str, file_path: str,
                       alias: str = 'latest', project: str = "nc_latent_tod",
                       entity: str = "kingb12") -> Any:
    api = wandb.Api()
    artifact: wandb.Artifact = api.artifact(f'{entity}/{project}/{artifact_name}:{alias}')
    download_path: str = artifact.download()
    with open(os.path.join(download_path, file_path), "r") as f:
        return json.load(f)


def read_jsonlines_artifact_to_df(artifact_name: str, file_path: str,
                       alias: str = 'latest', project: str = "nc_latent_tod",
                       entity: Optional[str] = None) -> pd.DataFrame:
    api = wandb.Api()
    entity = entity or os.environ.get("WANDB_ENTITY", "kingb12")
    artifact: wandb.Artifact = api.artifact(f'{entity}/{project}/{artifact_name}:{alias}')
    download_path: str = artifact.download()
    return pd.read_json(os.path.join(download_path, file_path), lines=True)


def read_run_artifact_logs(run_id: str, entity: Optional[str] = None, project: str = "nc_latent_tod", step: int = -1) -> Optional[List[DatasetTurnLog]]:
    """
    Get the running log associated with a wandb run id, if present
    :param run_id: run id (in url)
    :return: logs if an artifact of logs was added frmo the run
    """
    api = wandb.Api()
    entity = entity or os.environ.get("WANDB_ENTITY", "kingb12")
    run = api.run(f"{entity}/{project}/{run_id}")
    ignore_step: bool = step is None or step < 0
    for f in run.logged_artifacts():
        artifact_name, version = f.name.split(':')
        if (f.type == 'run_output' or f.type == 'running_log') and (ignore_step or artifact_name.endswith(str(step))):
            return read_json_artifact(f.name.split(':')[0],
                                      file_path="running_log.json",
                                      project=project,
                                      alias=f.version)


def read_all_artifacts_to_folder(run_id: str, output_path: str,  entity: Optional[str] = None, project: str = "nc_latent_tod"):
    api = wandb.Api()
    entity = entity or os.environ.get("WANDB_ENTITY", "kingb12")
    run = api.run(f"{entity}/{project}/{run_id}")
    for f in run.logged_artifacts():
        # we'll assume artifacts in the same run have unique file names
        f.download(output_path)


def get_running_logs_by_group(group_id: str, tags_in: List[str] = None, tags_not_in: List[str] = None,
                              project: str = "nc_latent_tod", entity: Optional[str] = None) -> List[List[DatasetTurnLog]]:
    """
    Return the running logs associated with a group from wandb, subject to tag filters.

    :param group_id: group to get runs from
    :param tags_in: only get runs from the group if tagged with one of these tags. Defaults to ["complete_run"]
    :param tags_not_in: ignore any run tagged with one of these tags. Defaults to ["outdated"]
    :return: running logs from matching runs
    """
    tags_in = tags_in or ["complete_run"]
    tags_not_in = tags_not_in or ["outdated"]
    api = wandb.Api()
    entity = entity or os.environ.get("WANDB_ENTITY", "kingb12")
    runs = api.runs(path=f"{entity}/{project}",
                    filters={"group": group_id, "tags": {"$in": tags_in, "$nin": tags_not_in}})
    result: List[List[DatasetTurnLog]] = []
    for run in runs:
        logs: List[DatasetTurnLog] = read_run_artifact_logs(run.id, project=project)
        result.append(logs)
    return result
