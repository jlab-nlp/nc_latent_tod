


import logging
from typing import List
import argparse

from nc_latent_tod.experiments.dataset_uploads import create_train_valid_dialogue_split, upload_from_manifest_group
from nc_latent_tod.experiments.dynamo_manifest import DynamoDBExperimentLogsManifest
from nc_latent_tod.experiments.manifest import ExperimentLogsManifest
from nc_latent_tod.schemas.data_types import ServiceSchema
from nc_latent_tod.schemas.reader import read_multiwoz_schema


def get_manifest(type: str, path: str) -> ExperimentLogsManifest:
    if type == "dynamodb":
        return DynamoDBExperimentLogsManifest(table_name=path)
    elif type == "json":
        return ExperimentLogsManifest(path)
    else:
        raise ValueError(f"Unsupported manifest type: {type}")

if __name__ == '__main__':    
    logging.basicConfig(level=logging.INFO)
    
    # parse arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--group-id", type=str, required=True, help="Group ID for items within the manifest to merge into a dataset")
    parser.add_argument("--manifest-path", type=str, help="Name of the DynamoDB table or path to the manifest file",
                        default="NCLatentTODExperimentsLogManifestTable")
    parser.add_argument("--validation-dialogues", type=int, help="Number of dialogues to use for validation", default=64)
    parser.add_argument("--manifest-type", type=str, help="Type of manifest file (e.g. 'dynamodb' or 'json')",
                        default="dynamodb")
    parser.add_argument("--hf-username", type=str, help="HF username for uploading datasets", required=True)
    parser.add_argument("--private", type=bool, help="Upload dataset privately (set to False for public)", default=True)   
    args = parser.parse_args()
    assert args.hf_username is not None, "Must provide a Hugging Face username"
    assert args.group_id, "Must provide a group ID"
    manifest: ExperimentLogsManifest = get_manifest(args.manifest_type, args.manifest_path)
    schema: List[ServiceSchema] = read_multiwoz_schema()
    
    dataset_name_or_path: str = upload_from_manifest_group(
        manifest=manifest,
        group_id=args.group_id,
        schema=schema,
        hf_username=args.hf_username,
        is_private_upload=args.private,
    )
    create_train_valid_dialogue_split(dataset_name_or_path,split_name="train", num_dialogues=args.validation_dialogues or 64)