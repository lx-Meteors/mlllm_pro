from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
import os
import random
from tqdm import tqdm
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default= "compressLLM_test_rajpurkar_squad_18",required=False, help='Directory including the configuration file')
    return parser.parse_args()


def get_examples_list(instruction_dataset_repo, split):
    instruction_dataset_name = instruction_dataset_repo.split('/')[-1]
    # cache long text for preventing full dataset traversal on each preparation.
    if os.path.exists(f'{instruction_dataset_name}_{split}_instruction_dataset.json'):
        with open(f'{instruction_dataset_name}_{split}_instruction_dataset.json', 'r', encoding='utf-8') as f:
            examples_list = json.load(f)
        return examples_list

    dataset = load_dataset(instruction_dataset_repo, split=split, streaming=True)

    examples_list = []
    for example in tqdm(dataset, desc="Processing examples"):
        examples_list.append(example)

    with open(f'{instruction_dataset_name}_{split}_instruction_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(examples_list, f, ensure_ascii=False)

    return examples_list


def get_ids(examples_list, tokenizer, min_len, split):
    examples = []
    minn = 9999
    maxn = 0
    for example in tqdm(examples_list, desc="Processing examples"):
        context_ids = tokenizer(example["context"], add_special_tokens=False)["input_ids"]
        question_ids = tokenizer(example["question"], add_special_tokens=False)["input_ids"]
        padding_token = tokenizer('\n', add_special_tokens=False)["input_ids"]

        input_ids = tokenizer("### Context:\n")["input_ids"] + context_ids \

        # 扩充到510
        if len(input_ids) < 510:
            input_ids = pad_sequence(input_ids,min_len,padding_token)
        else:
            continue
        inputs = torch.LongTensor(input_ids)
        lm_target =  torch.LongTensor(tokenizer("\n### Question:\n", add_special_tokens=False)["input_ids"]
                                      + question_ids + tokenizer("\n### Answer:\n")["input_ids"])
        examples.append({"input_ids": inputs, "lm_targets": lm_target})
    return examples

def pad_sequence(sequence, max_length, pad_value=0):
    res = pad_value * (max_length - len(sequence)) + sequence
    return res[-510:]


def get_examples(model_id, instruction_dataset_repo="sggetao/PwC", hf_token=None, token_num=1_000_000_000, min_len=512,
                 dataset_repo=None):
    model_name = model_id.split('/')[-1]
    instruction_dataset_name = instruction_dataset_repo.split('/')[-1]
    train_data_name = "train_" + model_name + "_data_name_" + instruction_dataset_name+ f"_len{min_len}_instruction.pt"
    eval_data_name = "eval_" + model_name + "_data_name_" + instruction_dataset_name+ f"_len{min_len}_instruction.pt"
    print(f"in:train_data_name:{train_data_name}")
    if os.path.exists(eval_data_name):
        print("loading data...")
        return torch.load(eval_data_name)
    print(f"preparing data :train_data_name:{train_data_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)


    # train_examples_list = get_examples_list(instruction_dataset_repo, split="train")
    test_examples_list = get_examples_list(instruction_dataset_repo, split="test")

    # train_data = get_ids(train_examples_list, tokenizer, min_len, split="train")
    test_data = get_ids(test_examples_list, tokenizer, min_len, split="test")

    # torch.save(train_data, train_data_name)
    torch.save(test_data, eval_data_name)

    return test_data


if __name__ == "__main__":
    args = parse_args()
    with open(args.work_dir + "/config.json") as f:
        config = json.load(f)

    training_config = config["training_config"]
    config["data_config"]["model_id"] = training_config["model_id"]

    print(config["data_config"])
    eval_examples = get_examples(**config["data_config"])
    print(len(eval_examples))
    print(eval_examples[50])

"""

unset HF_HUB_OFFLINE
HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0 python instruction_prepare_data.py --work_dir debug_CompressLLM_wo-cmp
HF_ENDPOINT=https://hf-mirror.com python instruction_prepare_data.py --work_dir debug_CompressLLM_wo-cmp
"""