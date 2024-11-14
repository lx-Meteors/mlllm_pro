import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
work_dir = "../compressLLM_baseline_merge_lora_rank-512_len-510_ratio-5_wo-ae"
with open(work_dir + f'/config.json') as f:
    config =json.load(f)

config["data_config"]["model_id"] = "../../../models/TinyLlama/TinyLlama_v1.1"
world_size = torch.cuda.device_count()
tokenizer = AutoTokenizer.from_pretrained(config["data_config"]["model_id"], token=config["data_config"]["hf_token"])

print("calculate BLEU4...")

if os.path.exists(work_dir + f'/instruction_cl_generate_text.json'):
    with open(work_dir + f'/instruction_cl_generate_text.json', 'r', encoding='utf-8') as f:
        examples_list =  json.load(f)



input_text = [entry["input_text"] for entry in examples_list]
cl_generate_text = [entry["cl_generate_text"] for entry in examples_list]


def cal_cl_token_acc(input_text, cl_generate_text, tokenizer):
    correct_tokens = 0
    total_tokens = 0
    acc = []
    bleus = []
    for input, decompress in tqdm(zip(input_text, cl_generate_text), desc="Processing examples", total=len(cl_generate_text)):
        cl_gen_text = decompress.replace("### Context:\n ", "", 1)
        cl_gen_ids = tokenizer(cl_gen_text, add_special_tokens=False)["input_ids"]
        input_ids = tokenizer(input, add_special_tokens=False)["input_ids"]

        total_tokens += len(cl_gen_ids)
        correct_tokens += sum(1 for o,d in zip(cl_gen_ids, input_ids) if o == d)
        acc.append(correct_tokens / total_tokens)
        correct_tokens = 0
        total_tokens = 0
        bleu4 = sentence_bleu([input_ids], cl_gen_ids, weights=(0.25, 0.25, 0.25, 0.25))
        bleus.append(bleu4)
    return np.mean(acc), np.mean(bleus)

acc, blue = cal_cl_token_acc(input_text, cl_generate_text, tokenizer)
print("acc:",acc)
print("bleu:",blue)