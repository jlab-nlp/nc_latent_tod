# Initial Self-Labelling Experiment

In this set of runs, we create an initial self-labelling of the MultiWOZ dataset using [bigcode/starcoder](https://huggingface.co/bigcode/starcoder) (15B).
This method follows the procedure described in sections 4.1-4.4 of [the paper](https://arxiv.org/abs/2404.15219).

## Overview

Each json file in this folder specifies a job for self-labelling 50 dialogues at a time.
The first job (`job_0.json`) needs to be run in isolation, labelling the first 50 dialogues and storing them in the manifest. 
The remaining can be run somewhat in parallel. 

> **Note:** In-context examples from each self-labelling **are retrieved from the manifest at job start**: if one ran all 160+ dependent jobs at once, each would only have 50 dialogues worth of in-context examples to retrieve from. In our original experiments reported in the paper, we used 3-5 parallel workers polling for jobs.

## Pre-requisite Setup

We used a couple of dependencies to handle managing jobs and their outputs. Open an Issue if you are having trouble setting these up or need help coming up with an alternative approach. These use Amazon AWS with low enough usage to likely fall in the free-tier.

### DynamoDB Table Setup

Outputs of individual jobs are written to a "manifest table", so that I can know where to retrieve in-context examples from, and eventually to publish a full self-labelled dataset.

I stored this table in DynamoDB. To replicate exactly as written (s), you need to:

1. Create a table called `NCLatentTODExperimentsLogManifestTable` in DynamoDB. Make sure `group_id` is the primary key, and `run_id` is the sort key. It is important to have both!
2. Configure AWS credentials for your runtime environment. I use the [environment variables method](https://docs.aws.amazon.com/sdkref/latest/guide/environment-variables.html).

**Alternatively**, you can use a local file as the manifest table. To do this, you can modify the configs in `cfg["manifest"]` with `local` as the type, and a file path as `manifest_path`.
 (re-run [generate_runs.py](./generate_runs.py) to modify all dependents accordingly)

### SQS Queue Setup

In lieu of more careful check-pointing, self-label jobs can be managed with a queue, where each job is a chunk of 50 dialogues. To facilitate this, you can:

1. Create a FIFO type SQS queue (low throughput is fine). I named mine: `nc-latent-tod-work-queue.fifo`.
2. When polling for jobs, make sure to set the queue name to the `NC_LATENT_TOD_QUEUE_PATH` environment variable:

```bash
export NC_LATENT_TOD_QUEUE_PATH="nc-latent-tod-work-queue.fifo"
python src/nc_latent_tod/experiments/poll_for_offline_label_jobs.py
```

## Inputs

- A 50 dialogue partition of the MultiWOZ 2.2 dataset (unlabelled conversations), processed turn by turn: [Brendan/multiwoz_turns_v22_partitioned](https://huggingface.co/datasets/Brendan/multiwoz_turns_v22_partitioned)
- Pre-trained LM [bigcode/starcoder](https://huggingface.co/bigcode/starcoder) (15B)
- An unsupervised retriever [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- A reference to a manifest, referencing any partitions we've already self-labelled.

## Outputs
- Job 0: A self-labelled partition of the 50 dialogues (**added to manifest** for use as in-context examples in future jobs)
- All jobs: complete self-labeled corpus

**The first job [job_0](./job_0.json) should run first, to completion. Then, the remaining jobs can be run in parallel to
speed up the process.**

## Steps to Reproduce

1. Run the first job ([job_0](./job_0.json)) to completion. This produces the initial self-labels for 50 dialogues, and makes them available to other jobs as in-context examples:

```bash
python src/nc_latent_tod/experiments/offline_labelling_experiment.py runs/offline_labelling_experiment/initial_labelling/job_0.json
```

Make sure your environment has access to the AWS resources described above, and is logged into Huggingface to use the repo-gated `bigcode/starcoder` model.

2. Run the remaining jobs (in [./dependents](./dependents)). The following will add all dependent jobs to the queue, and then start polling for them:

```bash
bash folder_runner.sh src/nc_latent_tod/experiments/enqueue_job.py runs/offline_labelling_experiment/initial_labelling/dependents
python src/nc_latent_tod/experiments/poll_for_offline_label_jobs.py
```

You can add additional workers which only poll:

```bash
python src/nc_latent_tod/experiments/poll_for_offline_label_jobs.py
```

If modifying configs to use file-based Manifests and queues, make sure the workers are both able to access the same copy of the file.

3. When all jobs are complete, you can merge them into a **Huggingface Dataset** like so:

```bash
python src/nc_latent_tod/experiments/upload_offline_labeled_dataset.py --group-id "initial_labelling" --hf-username "<YOUR_HUGGINFACE_USERNAME>"
```

By default, the dataset upload is private. This will upload a dataset named after the group, number of dialogues, and a hash of the manifest entries. See this example: [Brendan/manifest_self_labelled_initial_labelling_7800_FRnXLaU7I_EX](https://huggingface.co/datasets/Brendan/manifest_self_labelled_initial_labelling_7800_FRnXLaU7I_EX)


## Next Steps

1. To train a fine-tuned off-line labeller with this data, proceed to [runs/finetune_multitask/starcoder_3b/offline_label/README.md](../../finetune_multitask/starcoder_3b/offline_label/README.md). 
2. To train an end-to-end agent with this data, proceed to [runs/finetune_multitask/starcoder_3b/online_e2e/README.md](../../finetune_multitask/starcoder_3b/online_e2e/README.md). 