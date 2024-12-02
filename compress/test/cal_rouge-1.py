
from collections import Counter
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu

work_dir = "../compressLLM_random_instruction_(pre-train-multi-lora)_multi-lora_lm"
with open(work_dir + f'/config.json') as f:
    config =json.load(f)
config["data_config"]["model_id"] = "../../../models/TinyLlama/TinyLlama_v1.1"
world_size = torch.cuda.device_count()
tokenizer = AutoTokenizer.from_pretrained(config["data_config"]["model_id"], token=config["data_config"]["hf_token"])

print("calculate BLEU4...")

if os.path.exists(work_dir + f'/instruction_inference_results.json'):
    with open(work_dir + f'/instruction_inference_results.json', 'r', encoding='utf-8') as f:
        examples_list =  json.load(f)



input_text = [entry["answer"] for entry in examples_list]
cl_generate_text = [entry["generate"] for entry in examples_list]
def calculate_rouge_1(input_text, cl_generate_text):
    p = []
    r = []
    f = []
    for reference, generated in tqdm(zip(input_text, cl_generate_text), desc="Processing examples",total=len(cl_generate_text)):

        # 将参考摘要和生成摘要转换为单词列表（unigrams）
        ref_unigrams = reference.split()
        gen_unigrams = generated.split()

        # 计算参考摘要和生成摘要的 unigrams 频率
        ref_counter = Counter(ref_unigrams)
        gen_counter = Counter(gen_unigrams)

        # 计算匹配的 unigrams 数
        matching_unigrams = sum((ref_counter & gen_counter).values())  # 交集部分，即匹配的单词总数

        # 计算 Precision、Recall 和 F1
        precision = matching_unigrams / len(gen_unigrams) if len(gen_unigrams) > 0 else 0
        recall = matching_unigrams / len(ref_unigrams) if len(ref_unigrams) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        p.append(precision)
        r.append(recall)
        f.append(f1)

    return np.mean(p), np.mean(r), np.mean(f)

p, r, f = calculate_rouge_1(input_text, cl_generate_text)
print("precision:",p)
print("recall:",r)
print("f1:",f)