import copy
import json

from datasets import load_dataset, DatasetDict

if __name__ == '__main__':
    with open("job_0.json", 'r') as f:
        template = json.load(f)
    dataset_full: DatasetDict = load_dataset(template['data']['eval_set_path_or_name'], )
    split_names = sorted(list(k for k in dataset_full.keys() if k != template['data']['eval_set_split_name']))
    for split_name in split_names:
        run = copy.deepcopy(template)
        run['data']['eval_set_split_name'] = split_name
        run['manifest']['seed_retrievers_from_manifest'] = True
        run['manifest']['group_id_must_exist'] = True
        run['manifest']['manifest_must_exist'] = True
        run['dst']['retriever']['example_warmup'] = 0
        run['act_tag']['retriever']['example_warmup'] = 0
        run['data_warmup'] = 0
        with open(f"dependents/{split_name}.json", 'w') as f:
            json.dump(run, f, indent=4)

