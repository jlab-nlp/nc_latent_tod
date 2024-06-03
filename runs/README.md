# Run Configurations

This directory contains run configurations for all experiments. Each run configuration is a JSON file that
specifies the parameters for a given experiment type. 

## Offline Labelling Experiments

An offline labelling experiment config is compatible with 
[offline_labelling_experiment.py](../src/nc_latent_tod/experiments/offline_labelling_experiment.py). Given a dataset of 
un-labelled dialogues, it produces self-labels using an LLM for dialogue states and actions, and saves a dataset 
containing those labels.
