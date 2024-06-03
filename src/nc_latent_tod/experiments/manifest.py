import json
import logging
import os.path
from typing import TypedDict, Dict, List, Optional

from filelock import FileLock
from tqdm import tqdm

from nc_latent_tod.data_types import DatasetTurnLog
from nc_latent_tod.utils.artifacts import read_run_artifact_logs


class ManifestEntry(TypedDict):
    group_id: str  # the id of the group this run belongs to. Human generated but we should not re-do completed groups
    run_id: str  # wandb run id which generated the logs artifact
    num_logs: int  # number of logs in the artifact
    labelled_dataset_path_or_name: str  # what data did we label in this run
    labelled_dataset_split_name: str  # what data did we label in this run


# Group -> List of run_ids in the group
Manifest = Dict[str, List[ManifestEntry]]


class ExperimentLogsManifest:
    manifest_path: str
    lock: FileLock
    logs_cache_dir: Optional[str]

    def __init__(self, manifest_path: str, timeout: int = 5, create_if_not_exists: bool = False,
                 logs_cache_dir: Optional[str] = None) -> None:
        super().__init__()
        self.manifest_path = manifest_path
        self.lock = FileLock(manifest_path + ".lock", timeout=timeout)
        self.logs_cache_dir = logs_cache_dir
        if self.logs_cache_dir:
            os.makedirs(self.logs_cache_dir, exist_ok=True)
        with self.lock:
            if not os.path.exists(self.manifest_path):
                if create_if_not_exists:
                    with open(self.manifest_path, "w") as f:
                        json.dump({}, f)
                else:
                    raise FileNotFoundError(f"Manifest file {self.manifest_path} does not exist")
            elif os.path.getsize(self.manifest_path) == 0:
                with open(self.manifest_path, "w") as f:
                    f.write("{}")
            else:
                with open(self.manifest_path, "r") as f:
                    json.load(f)  # just verify it can be loaded

    @staticmethod
    def validate_entry(entry: ManifestEntry):
        # no blanks allowed, has to have some logs
        assert entry["group_id"], entry['group_id']
        assert entry["run_id"], entry['run_id']
        assert entry["num_logs"] > 0, entry['num_logs']
        assert entry["labelled_dataset_path_or_name"], entry['labelled_dataset_path_or_name']
        assert entry["labelled_dataset_split_name"], entry['labelled_dataset_split_name']

    def add_entry(self, entry: ManifestEntry):
        """
        Add an entry to the manifest. Will create the manifest file if it does not exist.
        :param entry: entry to add
        """
        self.validate_entry(entry)
        with self.lock:
            with open(self.manifest_path, "r") as f:
                manifest: Manifest = json.load(f)
            group_id = entry["group_id"]
            if group_id not in manifest:
                manifest[group_id] = []
            if any(e["run_id"] == entry["run_id"] for e in manifest[group_id]):
                logging.error(f"Run {entry['run_id']} already in group {group_id}, Skipping!")
                return
            manifest[group_id].append(entry)
            with open(self.manifest_path, "w") as f:
                json.dump(manifest, f)

    def read_group(self, group_id: str) -> List[ManifestEntry]:
        """
        Read all entries for a particular group
        :param group_id: group id to read
        :return: list of entries for the group
        """
        with self.lock:
            with open(self.manifest_path, "r") as f:
                manifest: Manifest = json.load(f)
            return manifest[group_id]

    def get_logs_from_group(self, group_id: str) -> List[DatasetTurnLog]:
        """
        Given the group id, read all logs from the group. For example, say we are self-labelling all of the MultiWOZ
        training set dialogues in jobs of 50 dialogues each. Job results might be added to manifest in a group with a
        group id. This function reads all such job results and returns them as a unified list.

        :param group_id: id of the group to read logs from
        :return: merged list of logs
        """
        entries: List[ManifestEntry] = self.read_group(group_id)
        logging.info(f"Found {len(entries)} entries for group {group_id}")
        all_logs: List[DatasetTurnLog] = []
        for entry in tqdm(entries, desc="Reading logs from w&b artifacts"):
            if self.logs_cache_dir:
                cache_path = os.path.join(self.logs_cache_dir, entry['run_id'] + ".json")
                if os.path.exists(cache_path):
                    logging.info(f"Reading logs from cache for run {entry['run_id']}")
                    run_logs: List[DatasetTurnLog] = json.load(open(cache_path, 'r'))
                    if not len(run_logs) == entry['num_logs']:
                        # invalid cache hit: re-download
                        logging.error(f"Invalid cache at {cache_path}:"
                                      f"Expected {entry['num_logs']} logs but found {len(run_logs)}")
                    else:
                        # valid cache hit
                        all_logs.extend(run_logs)
                        continue
            # cache miss or not caching
            logging.info(f"Reading logs from run {entry['run_id']}, which "
                         f"labelled {entry['num_logs']} turns in {entry['labelled_dataset_path_or_name']}, "
                         f"split={entry['labelled_dataset_split_name']}")
            run_logs: List[DatasetTurnLog] = read_run_artifact_logs(entry["run_id"])
            assert len(run_logs) == entry['num_logs'], f"Expected {entry['num_logs']} logs but found {len(run_logs)}"
            all_logs.extend(run_logs)
            if self.logs_cache_dir:
                cache_path = os.path.join(self.logs_cache_dir, entry['run_id'] + ".json")
                logging.info(f"Caching logs for run {entry['run_id']} at {cache_path}")
                json.dump(run_logs, open(cache_path, 'w'))
        return all_logs

    def remove_entry(self, group_id: str, run_id: str) -> None:
        """
        Remove an entry from the manifest
        :param group_id: group id to remove from
        :param run_id: run id to remove
        """
        with self.lock:
            with open(self.manifest_path, "r") as f:
                manifest: Manifest = json.load(f)
            entries: List[ManifestEntry] = manifest[group_id]
            manifest[group_id] = [e for e in entries if e["run_id"] != run_id]
            with open(self.manifest_path, "w") as f:
                json.dump(manifest, f)
