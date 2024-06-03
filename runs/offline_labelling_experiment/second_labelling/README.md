# Re-labeling with a fine-tuned labeler

In this step, we use the fine-tuned [StarCoder 3B](https://huggingface.co/bigcode/starcoderbase-3b) as a Dialogue State Tracker and Dialogue Act Tagger to re-label the corpus with improved pseudo-labels

## Overview

[job_0.json](job_0.json) specifies a configuration for the [Offline Label Experiment](/src/nc_latent_tod/experiments/offline_labelling_experiment.py) experiment, covering the first 50 dialogues. You can edit parameters in this one, and re-generate the dependent jobs using `python generate_runs.py` from this directory.

**IF YOU ARE USING YOUR OWN FINE-TUNED MODEL, YOU NEED TO REPLACE THIS IN THE CONFIG:**

```bash
sed -i '' 's|Brendan/tod-zero-bqag3oyb-32000|your-data-set-path-here|g' offline_label.json
```

## Inputs

- A fine-tuned offline labeller, such as [Brendan/tod-zero-bqag3oyb-32000](https://huggingface.co/Brendan/tod-zero-bqag3oyb-32000)
- The unlabelled corpus of interest: [Brendan/multiwoz_turns_v22_partitioned](https://huggingface.co/datasets/Brendan/multiwoz_turns_v22_partitioned)

## Outputs

- A completely pseudo-labelled version of the corpus

## Steps to Reproduce

1. Make sure the pre-requisite setup for [initial labelling (SQS, Dynamo)](../initial_labelling/README.md) has been completed.

2. Since fine-tuned models use no in-context examples or shared example pool, you can run the jobs all at once or in any order. Run the first job ([job_0](./job_0.json)) via:

```bash
python src/nc_latent_tod/experiments/offline_labelling_experiment.py runs/offline_labelling_experiment/second_labelling/job_0.json
```

3. Run the remaining jobs (in [./dependents](./dependents)). The following will add all dependent jobs to the queue, and then start polling for them:

```bash
bash folder_runner.sh src/nc_latent_tod/experiments/enqueue_job.py runs/offline_labelling_experiment/second_labelling/dependents
python src/nc_latent_tod/experiments/poll_for_offline_label_jobs.py
```

You can add additional workers which only poll:

```bash
python src/nc_latent_tod/experiments/poll_for_offline_label_jobs.py
```

If modifying configs to use file-based Manifests and queues, make sure the workers are all able to access the same copy of the file.

3. When all jobs are complete, you can merge them into a **Huggingface Dataset** like so:

```bash
python src/nc_latent_tod/experiments/upload_offline_labeled_dataset.py --group-id "second_labelling" --hf-username "<YOUR_HUGGINFACE_USERNAME>"
```

By default, the dataset upload is private. This will upload a dataset named after the group, number of dialogues, and a hash of the manifest entries.

## Next Steps

1. To train a fine-tuned off-line labeller with this data (we did not try this in the paper, after a second labelling), proceed to [runs/finetune_multitask/starcoder_3b/offline_label/README.md](../../finetune_multitask/starcoder_3b/offline_label/README.md). 
2. To train an end-to-end agent with this data, proceed to [runs/finetune_multitask/starcoder_3b/online_e2e/README.md](../../finetune_multitask/starcoder_3b/online_e2e/README.md). 