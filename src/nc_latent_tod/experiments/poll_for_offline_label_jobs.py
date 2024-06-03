import dataclasses
import logging
import os
import pprint

import wandb
from wandb.sdk.lib.runid import generate_id

from nc_latent_tod.experiments.config import OfflineLabellingLMExperimentConfig
from nc_latent_tod.experiments.offline_labelling_experiment import main as offline_labelling_main
from nc_latent_tod.experiments.utils import verify_cuda_is_available_if_needed, parse_experiment_config, \
    ModelCachingModuleBuilder
from nc_latent_tod.utils.sqs_queue import SQSQueue

if __name__ == '__main__':
    """
    Given a queue specified by `NC_LATENT_TOD_QUEUE_PATH`, dequeue jobs and run them as offline labelling experiments. Running with this pattern
    allows continuous labelling and doesn't require re-loading the model each time. 
    """
    logging.basicConfig(level=logging.INFO)
    verify_cuda_is_available_if_needed()
    builder: ModelCachingModuleBuilder = ModelCachingModuleBuilder()
    queue_path: str = os.environ['NC_LATENT_TOD_QUEUE_PATH']
    work_queue: SQSQueue = SQSQueue(queue_path)
    next_job = work_queue.dequeue()
    i = 1
    while next_job is not None:
        job_path, job_obj = next_job
        print(f"Starting {i}th job of {len(work_queue)} at {job_path}:\n\n {pprint.pformat(job_obj)}\n\n")
        run_id: str = generate_id()
        cfg: OfflineLabellingLMExperimentConfig = parse_experiment_config(
            config_data=job_obj,
            config_path=job_path,
            data_class=OfflineLabellingLMExperimentConfig,
            run_id=run_id
        )
        wandb.init(
            config=dataclasses.asdict(cfg),
            project="nc_latent_tod", entity=os.environ.get("WANDB_ENTITY", "kingb12"),
            name=cfg.wandb.run_name,
            notes=cfg.wandb.run_notes,
            group=cfg.wandb.run_group,
            tags=cfg.wandb.run_tags,
            id=run_id
        )
        offline_labelling_main(cfg, builder=builder)
        wandb.finish()
        i += 1
        next_job = work_queue.dequeue()
    print("All jobs complete")
