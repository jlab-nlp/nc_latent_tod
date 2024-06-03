import os
from typing import Optional, List

import boto3
from botocore.exceptions import ClientError
import logging

from nc_latent_tod.experiments.manifest import ExperimentLogsManifest, ManifestEntry


def clean_entry(entry: dict) -> ManifestEntry:
    return {
        "group_id": entry["group_id"],
        "run_id": entry["run_id"],
        "num_logs": int(entry["num_logs"]),
        "labelled_dataset_path_or_name": entry["labelled_dataset_path_or_name"],
        "labelled_dataset_split_name": entry["labelled_dataset_split_name"]
    }


class DynamoDBExperimentLogsManifest(ExperimentLogsManifest):
    def __init__(self, aws_region: str = "us-west-2", logs_cache_dir: Optional[str] = None,
                 table_name: str = 'NCLatentTODExperimentsLogManifestTable', **kwargs):
        self.dynamodb = boto3.resource('dynamodb', region_name=aws_region)
        self.table = self.dynamodb.Table(table_name)
        self.logs_cache_dir = logs_cache_dir
        if self.logs_cache_dir:
            os.makedirs(self.logs_cache_dir, exist_ok=True)

    def add_entry(self, entry: ManifestEntry):
        self.validate_entry(entry)
        try:
            self.table.put_item(Item=entry)
        except ClientError as e:
            logging.error(f"Failed to add entry to DynamoDB: {e}")

    def read_group(self, group_id: str) -> List[ManifestEntry]:
        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('group_id').eq(group_id)
            )
            return [clean_entry(e) for e in response['Items']]
        except ClientError as e:
            logging.error(f"Failed to read group from DynamoDB: {e}")
            return []

    def remove_entry(self, group_id: str, run_id: str) -> None:
        try:
            self.table.delete_item(
                Key={
                    'group_id': group_id,
                    'run_id': run_id
                }
            )
        except ClientError as e:
            logging.error(f"Failed to remove entry from DynamoDB: {e}")


if __name__ == "__main__":
    manifest = DynamoDBExperimentLogsManifest()
    manifest.read_group('mar_13')
    manifest.add_entry({
        "group_id": "mock_group",
        "run_id": "mock_run",
        "num_logs": 1,
        "labelled_dataset_path_or_name": "mock_path",
        "labelled_dataset_split_name": "mock_split"
    })
    group = manifest.read_group('mock_group')
    print(group)
