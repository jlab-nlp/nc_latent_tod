import logging
import os
import sys
from typing import List

from nc_latent_tod.utils.general import read_json
from nc_latent_tod.utils.sqs_queue import SQSQueue

# Sometimes I set this if I want to enqueue a whole folder at once, minus a few that completed
SKIP_PATHS: List[str] = []


if __name__ == '__main__':
    """
    Given a path to a JSON file, read the object and enqueue it at the specified queue path (`NC_LATENT_TOD_QUEUE_PATH` env variable).
    """
    queue_path = os.environ['NC_LATENT_TOD_QUEUE_PATH']
    job_queue: SQSQueue = SQSQueue(queue_path)
    path: str = sys.argv[1]
    json_obj = read_json(sys.argv[1])
    # enqueue path with its contents
    if path in SKIP_PATHS:
        logging.warning(f"Skipping {path}, aleady covered!")
        sys.exit(0)
    else:
        job_queue.enqueue((path, json_obj))
        print(f"Enqueued {path} to {queue_path}")
