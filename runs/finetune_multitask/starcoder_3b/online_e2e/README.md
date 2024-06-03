# Fine-tuning the final end-to-end agent

In this step, we fine-tune [StarCoder 3B](https://huggingface.co/bigcode/starcoderbase-3b) as an end-to-end dialogue system, composed of a dialogue state tracker, policy, and response generator.

## Overview

[online_e2e.json](online_e2e.json) specifies a configuration for the [Finetune Multitask](/src/nc_latent_tod/peft_finetune/finetune_multitask.py) experiment.

**IF YOU ARE USING A DATASET YOU CREATED WITH YOUR OWN INITIAL LABELING, YOU NEED TO ADD THIS TO THE CONFIG AT `data.train_set_path_or_name` AND `data.eval_set_path_or_name`:**

```bash
sed -i '' 's|Brendan/manifest_self_labelled_tod_zero_bqag3oyb_8324_htb95VGQDY1x|your-data-set-path-here|g' online_e2e.json
```

## Inputs

- A self-labelled dataset, for example [Brendan/manifest_self_labelled_tod_zero_bqag3oyb_8324_htb95VGQDY1x](https://huggingface.co/datasets/Brendan/manifest_self_labelled_tod_zero_bqag3oyb_8324_htb95VGQDY1x)
- A pre-trained code-aware LLM (we use [StarCoder 3B](https://huggingface.co/bigcode/starcoderbase-3b), others can be configured)

## Outputs

- A series of model checkpoints, with evaluation scores logged using **held out self-labels**. Caller must look over the checkpoints and pick which to upload to Huggingface, before continuing with steps in [runs/online_e2e_experiment/test_set/](/runs/online_e2e_experiment/test_set/README.md)

## Steps to Reproduce

1. After editing the config to use the appropriate dataset (see above) run the experiment:

```bash
python src/nc_latent_tod/peft_finetune/finetune_multitask.py runs/finetune_multitask/starcoder_3b/online_e2e/online_e2e.json
```

2. Pick a checkpoint using the evaluation scores, and upload the model to Huggingface. I used the [Huggingface CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

## Next Steps

Evaluate the agent! [See this example]([runs/online_e2e_experiment/test_set/](/runs/online_e2e_experiment/test_set/README.md))