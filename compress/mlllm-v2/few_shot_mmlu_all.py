from collections import defaultdict

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
    parser.add_argument('--work_dir', type=str, default= "test",required=False, help='Directory including the configuration file')
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
        example['choices'] = ("(A)." + example['choices'][0] + "\n" +
                              "(B)." + example['choices'][1] + "\n" +
                              "(C)." + example['choices'][2] + "\n" +
                              "(D)." + example['choices'][3] + "\n")
        if example['answer'] == 0:
            example['answer'] = 'A'
        elif example['answer'] == 1:
            example['answer'] = 'B'
        elif example['answer'] == 3:
            example['answer'] = 'C'
        else:
            example['answer'] = 'D'
        examples_list.append(example)

    with open(f'{instruction_dataset_name}_{split}_instruction_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(examples_list, f, ensure_ascii=False)

    return examples_list


def get_ids(dev_data, instruction_dataset_name, examples_list, tokenizer, min_len, split):
    examples = []
    re_examples_list = []
    minn = 9999
    maxn = 0
    for example in tqdm(examples_list, desc="Processing examples"):
        few_shot_5 = create_5_shot_prompt(example["subject"], dev_data)
        prompt = "Which one in ABCD is correct?"
        question_ids = tokenizer(example["question"], add_special_tokens=False)["input_ids"]
        choices_ids = tokenizer(example["choices"], add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(example["answer"], add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        all_prompt_ids = tokenizer("### Context:\n")["input_ids"] + tokenizer(few_shot_5)["input_ids"] + \
                         tokenizer("### Question:\n")["input_ids"] + question_ids + \
                         tokenizer("\n")["input_ids"] + choices_ids

        all_response_ids = tokenizer("\n### Prompt:\n", add_special_tokens=False)["input_ids"]+ prompt_ids +\
                           tokenizer("\n### Answer:\n", add_special_tokens=False)["input_ids"]
        padding_token = tokenizer('\n', add_special_tokens=False)["input_ids"]
        # # 生成 instruction_target，它是一个标签，用来指导模型学习预测目标
        if len(all_prompt_ids) < 510:
            all_prompt_ids = pad_sequence(all_prompt_ids,min_len,padding_token)
        else:
            continue
        instruction_target = [-100 for x in all_prompt_ids] + [x for x in all_response_ids]
        instruction_target = instruction_target[1:]
        if split == 'train':
            all_ids = all_prompt_ids + all_response_ids
        else:
            all_ids = all_prompt_ids

        minn = min(minn, len(all_ids))
        maxn = max(maxn, len(all_ids))

        inputs = torch.LongTensor(all_ids)
        # 如果是训练的时候？
        lm_target = torch.LongTensor(all_response_ids)

        instruction_target = torch.LongTensor(instruction_target)

        if split == "test":
            examples.append({"input_ids": inputs, "lm_targets": lm_target})
        else:
            examples.append({"input_ids": inputs, "lm_targets": lm_target,
                             "instruction_target": instruction_target})
        re_examples_list.append(example)
    with open(f'{instruction_dataset_name}_{split}_instruction_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(re_examples_list, f, ensure_ascii=False)
    return examples

def create_5_shot_prompt(subject, dev_data):
    data = dev_data.get(subject)
    # 选择前 5 个问题进行 5-shot 提示
    index = 1
    prompt = f"Task: Solve the following {subject} problems.\n\n"
    example = ""
    for item in data:
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]
        example += f"Example {index} :\n" + "### Question:" + question + "\n" + \
                   choices + "\n### Answer:\n" + answer + "\n"
        index += 1
    prompt = prompt + example + "Now, solve the following:\n"
    return prompt


def get_ids_dev(dev_examples_list):
    grouped_data = {}

    for item in dev_examples_list:
        subject = item['subject']
        if subject not in grouped_data:
            grouped_data[subject] = []
        grouped_data[subject].append(item)
    return grouped_data

def pad_sequence(sequence, max_length, pad_value=0):
    """
    将序列填充到指定长度。
    :param sequence: 原始序列（List[int]）
    :param max_length: 目标长度
    :param pad_value: 填充值（默认为0）
    :return: 填充后的序列
    """
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
        return torch.load(eval_data_name), torch.load(eval_data_name)
    print(f"preparing data :train_data_name:{train_data_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)


    dev_examples_list = get_examples_list(instruction_dataset_repo, split="train")
    test_examples_list = get_examples_list(instruction_dataset_repo, split="test")

    dev_data = get_ids_dev(dev_examples_list)
    test_data = get_ids(dev_data, instruction_dataset_name, test_examples_list, tokenizer, min_len, split="test")

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