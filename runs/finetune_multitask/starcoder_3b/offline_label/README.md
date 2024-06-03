# Fine-tuning an offline labeler

In this step, we fine-tune [StarCoder 3B](https://huggingface.co/bigcode/starcoderbase-3b) as a Dialogue State Tracker and Dialogue Act Tagger using our initial self-labels in order to produce higher-quality pseudo-labels for the unlabeled training corpus.

## Overview

[offline_label.json](offline_label.json) specifies a configuration for the [Finetune Multitask](/src/nc_latent_tod/peft_finetune/finetune_multitask.py) experiment.

**IF YOU ARE USING A DATASET YOU CREATED WITH YOUR OWN INITIAL LABELING, YOU NEED TO ADD THIS TO THE CONFIG AT `data.train_set_path_or_name` AND `data.eval_set_path_or_name`:**

```bash
sed -i '' 's|Brendan/manifest_self_labelled_mar_18_8324__N3_otj9Am5v|your-data-set-path-here|g' offline_label.json
```

## Inputs

- A self-labelled dataset, for example [Brendan/manifest_self_labelled_mar_18_8324__N3_otj9Am5v](https://huggingface.co/datasets/Brendan/manifest_self_labelled_initial_labelling_7800_FRnXLaU7I_EX)
- A pre-trained code-aware LLM (we use [StarCoder 3B](https://huggingface.co/bigcode/starcoderbase-3b), others can be configured)

## Outputs

- A series of model checkpoints, with evaluation scores logged using **held out self-labels**. Caller must look over the checkpoints and pick which to upload to Huggingface, before continuing with steps in [runs/offline_labelling_experiment/second_labelling/README.md](../../../offline_labelling_experiment/second_labelling/README.md)

## Steps to Reproduce

1. After editing the config to use the appropriate dataset (see above) run the experiment:

```bash
python src/nc_latent_tod/peft_finetune/finetune_multitask.py runs/finetune_multitask/starcoder_3b/offline_label/offline_label.json
```

2. Pick a checkpoint using the evaluation scores, and upload the model to Huggingface. I used the [Huggingface CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)