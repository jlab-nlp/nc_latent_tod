# Self-Labelled MultiWOZ Dataset

Manifest Group: `{group_id}`
Number of Dialogues: `{num_dialogues}`
Number of Turns: `{num_turns}`

This dataset was created via a self-labelling process composed of multiple runs, in which un-labelled dialogue data
(utterances from the user and system only) is labelled with pseudo-annotations for the belief state,
system acts, and de-lexicalized system response. 

Here is the list of W&B runs contributing to this dataset. Each run is a self-labelling run, and should be well defined
as partitioning a larger dataset in to chunks labelled with the same process. Consult each run to validate the git 
revision and configuration parameters.

{wandb_run_link_list}

Here is the configuration of the 'first' job in this manifest group, if found.

```json
{config}
```

The dataset was created from the following manifest entries:

```json
{manifest_entries}
```
