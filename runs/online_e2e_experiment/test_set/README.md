# Online E2E Experiment (Final Evaluation)

## Overview

[step-2-final.json](step-2-final.json) specifies a configuration for [online_e2e_experiment.py](/src/nc_latent_tod/experiments/online_e2e_experiment.py), which performs inference on an end-to-end dialogue system and reports evaluation scores. Currently, this specifies our final end-to-end model on the test set

**TO USE A DIFFERENT SPLIT:** see the [available splits](https://huggingface.co/datasets/Brendan/multiwoz_turns_v22) and modify `data.eval_set_split_name`. 

**TO USE A DIFFERENT MODEL:** replace the model name everywhere in the config:

```bash
sed -i '' 's|Brendan/nc-latent-tod-step-2-final|your-model-name-or-path-here|g' online_e2e.json
```

## Inputs

- A model fine-tuned in the process described in [finetune_multitask/starcoder_3b/online_e2e](../../finetune_multitask/starcoder_3b/online_e2e/README.md)

## Outputs

- Evaluation scores (printed, logged to W&B)

## Steps to Reproduce

1. After editing the config to use the appropriate model/dataset split (see above) run the experiment:

```bash
python src/nc_latent_tod/experiments/online_e2e_experiment.py runs/online_e2e_experiment/test_set/step-2-final.json
```
