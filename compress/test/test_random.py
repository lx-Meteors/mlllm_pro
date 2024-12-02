
import json
import re

import numpy
import numpy as np
import matplotlib.pyplot as plt
import os

import torch

#
# lm_loss = []
# compress_loss = []
# with open('compressLLM_instruction_baseline_cl&lm_1-1/instruction_info.json', 'r') as f:
#     data = json.load(f)
#     for run in data:
#         lm_loss.append(run['training_loss']['lm_loss'])
#         compress_loss.append(run['training_loss']['compress_loss'])
# avg_lm_loss = np.mean(lm_loss)
# avg_compress_loss = np.mean(compress_loss)
# print("avg_lm_loss：",avg_lm_loss)
# print("avg_compress_loss：",avg_compress_loss)

# 由于无法使用nltk库进行自动词性标注，我将手动标注示例文本的词性

# # 示例文本和手动标注的词性
# text = "The quick brown fox jumps over the lazy dog."
# tagged_tokens_manual = [("The", "DT"), ("quick", "JJ"), ("brown", "JJ"),
#                         ("fox", "NN"), ("jumps", "VB"), ("over", "IN"),
#                         ("the", "DT"), ("lazy", "JJ"), ("dog", "NN"), (".", ".")]
#
# # 重新定义词性颜色映射
# pos_color_map = {
#     'DT': 'purple',  # 限定词
#     'JJ': 'green',   # 形容词
#     'NN': 'red',     # 名词
#     'VB': 'blue',    # 动词
#     'IN': 'orange',  # 介词或从属连词
#     '.': 'black'     # 句号
# }
#
#
# # 自动生成HTML代码，根据标注的词性设置颜色
#
# html_output = ""
#
# for word, tag in tagged_tokens_manual:
#     color = pos_color_map.get(tag, 'black')
#     html_output += f'<span style="color:{color};">{word}</span> '
#
# print(html_output)


# import torch
#
# # 假设 logits 是一个形状为 (510, 32000) 的 tensor
# logits = torch.randn(510, 32000)  # 示例 logits，实际应用中应为你的模型输出
#
# # 获取每个 token 最可能的词汇索引
# predicted_indices = torch.argmax(logits, dim=-1)
#
# # 转换为 Python 列表或其他需要的格式
# predicted_indices_list = predicted_indices.tolist()  # 将 tensor 转为列表
#
# # 如果只想要第一个 token 的预测
# first_token_index = predicted_indices.item()  # 只获取第一个 token 的预测索引









#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class TripleLoraLayer(nn.Module):
#     def __init__(self, in_features, out_features, r_cl=8, r_lm=8, r_cl_prime=8, alpha=1.0, weight=None):
#         super(TripleLoraLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha  # Scaling factor
#
#         # 原始权重矩阵 W
#         self.weight = nn.Parameter(weight if weight is not None else torch.randn(out_features, in_features))
#
#         # 压缩 token 的 LoRA 模块参数 (A_cl, B_cl)
#         self.lora_A_cl = nn.Parameter(torch.randn(in_features, r_cl))
#         self.lora_B_cl = nn.Parameter(torch.randn(r_cl, out_features))
#
#         # LLM token 的 LoRA 模块参数 (A_lm, B_lm)
#         self.lora_A_lm = nn.Parameter(torch.randn(in_features, r_lm))
#         self.lora_B_lm = nn.Parameter(torch.randn(r_lm, out_features))
#
#         # 额外的压缩 token 的 LoRA 模块参数 (A_cl', B_cl')
#         self.lora_A_cl_prime = nn.Parameter(torch.randn(in_features, r_cl_prime))
#         self.lora_B_cl_prime = nn.Parameter(torch.randn(r_cl_prime, out_features))
#
#         # 初始化 LoRA 参数
#         nn.init.kaiming_uniform_(self.lora_A_cl, a=5 ** 0.5)
#         nn.init.zeros_(self.lora_B_cl)
#         nn.init.kaiming_uniform_(self.lora_A_lm, a=5 ** 0.5)
#         nn.init.zeros_(self.lora_B_lm)
#         nn.init.kaiming_uniform_(self.lora_A_cl_prime, a=5 ** 0.5)
#         nn.init.zeros_(self.lora_B_cl_prime)
#
#     def forward(self, x, token_type):
#         # 原始权重的计算结果
#         result = F.linear(x, self.weight)
#
#         if token_type == 'compress':
#             # 压缩 token 的计算
#             result += self.alpha * (x @ self.lora_A_cl @ self.lora_B_cl)
#         elif token_type == 'lm':
#             # LLM token 的计算
#             result += self.alpha * (x @ self.lora_A_lm @ self.lora_B_lm)
#         elif token_type == 'compress_prime':
#             # 额外压缩 token 的计算
#             result += self.alpha * (x @ self.lora_A_cl_prime @ self.lora_B_cl_prime)
#
#         return result

import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch import cosine_similarity
from tqdm import tqdm

# # 假设 bsz, seq_len, mem_size, emb_size 都已定义
# bsz, seq_len, mem_size, emb_size = 2, 5, 3, 4  # 示例值
# # 创建 seq_len 部分为 1 的张量，形状为 (bsz, seq_len, emb_size)
# ones_part = torch.ones(bsz, seq_len, emb_size)
#
# # 创建 mem_size 部分为 0 的张量，形状为 (bsz, mem_size, emb_size)
# zeros_part = torch.zeros(bsz, mem_size, emb_size)
#
# # 拼接这两个张量，形成最终的形状 (bsz, seq_len + mem_size, emb_size)
# result = torch.cat([ones_part, zeros_part], dim=1)
#
# print(result)
# 创建一个形状为 (1, 612, 2048) 的矩阵，前 306 行是 0，后 306 行是 1

# lm_loss = []
# compress_loss = []
# use_compress_loss = False
# with open('compressLLM_multi_lora_510_ratio_lm&cl/instruction_info.json', 'r') as f:
#     data = json.load(f)
#     for run in data:
#         lm_loss.append(run['training_loss']['lm_loss'])
#         if 'compress_loss' in run['training_loss']:
#             use_compress_loss = True
#             compress_loss.append(run['training_loss']['compress_loss'])
# avg_lm_loss = np.mean(lm_loss)
# compress_loss = np.mean(compress_loss)
# print(avg_lm_loss)
# print(compress_loss)
#
# mask = {"a":1,"b":2}
# if "a" in mask:
#     print(6)
from transformers import AutoTokenizer, AutoModelForCausalLM

#
# # 选择一个模型的tokenizer（如BERT模型）
# work_dir = "../compressLLM_multi_lora_510_ratio_lm&cl"
# with open(work_dir + f'/config.json') as f:
#     config =json.load(f)
#
# config["data_config"]["model_id"] = "../../../models/TinyLlama/TinyLlama_v1.1"
# world_size = torch.cuda.device_count()
# tokenizer = AutoTokenizer.from_pretrained(config["data_config"]["model_id"], token=config["data_config"]["hf_token"])
# model = AutoModelForCausalLM.from_pretrained(config["data_config"]["model_id"], token=config["data_config"]["hf_token"])
#
#
# correct_tokens = 0
# # 输入文本
# # 41
# # text1 = "Hindus, who urged disciplining of a Northern Ireland Catholic priest who reportedly linked yoga to Satan, are disheartened over the Church’s silence on the issue.\n\nFather Reverend Roland Colhoun, a priest at Waterside Parish of Roman Catholic Diocese of Derry in Northern Ireland, as reported by Derry Journal, “warned parishioners against taking part in yoga” while saying mass in Drumsurn recently. “Yoga is certainly a risk. There’s the spiritual health risk”, Journal quoted him.\n\nUK’s The Independent quoted him as saying “Yoga leads to Satan” and that he fears it could lead to “The Kingdom of Darkness”. “It’s a slippery slope from yoga to Satan”, RT channel said quoting him.\n\nHindu statesman Rajan Zed, in a statement in Nevada (USA) today, said that reports suggested that neither the Vatican and nor the Diocese of Derry Bishop, Most Reverend Donal McKeown, had offered apology on the actions of the priest who just trashed a highly revered and respected ancient practice.\n\nZed, who is President of Universal Society of Hinduism, stressed that although introduced and nourished by Hinduism, yoga was a world heritage and liberation powerhouse being utilized by millions of people of various faith traditions worldwide for achieving physical and mental fitness.\n\nWhy this Derry priest was unnecessarily disparaging yoga whom Patanjali described as “a methodical effort to attain perfection” and about whom US National Institutes of Health indicated: yoga may help one to feel more relaxed, be more flexible, improve posture, breathe deeply, and get rid of stress? Rajan Zed asked.\nZed further said that Bishop McKeown, whose Diocese of Derry was directly responsible for the actions of this priest, should apologize to the hurt worldwide yoga community and clearly state where he stood on this issue. Moreover, Vatican should also clarify its stand on yoga, honestly and transparently.\nYoga, referred as “a living fossil”, was a mental and physical discipline, for everybody to share and benefit from, whose traces went back to around 2,000 BCE to Indus Valley civilization, Rajan Zed pointed out and added that yoga was the repository of something basic in the human soul and psyche.",
# # text2 = "Hindus, who urged disining of a Ireland Ireland religious Catholic who reportedly linked ygs to thean, are dis heartened over the Church’s silence on the issue.\n\nFather ReverendR Colhoun, a W at Waterside Parish of Roman Catholicedese of ofry in Northern Ireland, as reported by Denry Journal, “Iedionioners against taking part in y in” while saying mass in Drsurn recently. “Y has is certainly a risk. There’s the religious health risk”, Journal quoted him.\n\nStep’s The/ quoted him as saying “Y leads leads to satan” and that he fears it could lead to “The Kingdom of ofness”. “It’s a slipperyys from y\" to (an”, RT channel said quoting him.\n\nH Muslim statesman Rajan Zed, in a statement in Nevada ( USA) today, said that reports suggested that neither the Vatican and or theineese of derry,, Most Reverend Donal Mc Mown, had offered apology on the actions of the Francis who just tr\n a highly ored and respected practice practice.\n\nZed, who is President of the Society of H toism, stressed that although introduced and nourished by H byism, y\" was a world herity and releasedation powerhouse being utilized by millions of people of various faith traditions worldwide for’ieving physical and mental fitness.\n\nWhy this derry it wasitarilyagesaging y — whom Patanjali described as “a methodical effort to attain perfection” and about whom US Nationalom US of Health indicated: y  may help one to feel more relaxed, be more our, improve posture, breathe she, and get rid of stress? Rajan Zed asked.\n\nZed further said that Church Mckeown, whoseThese of derry was directly responsible for the actions of this apolog, should apologize to the hurt worldwide y community community and clearly state where he stood on this issue. issue, Vatican should also' its stand on y In, honestly and transparently.\n\nYoga, referring as “a living fossil”, was a mental and physical and, for everybody to share and benefit from, whose whose went back to around 2 ### Context:\n for everybody to share and benefit from, whose traces went back to"
# text1 = "2017 has been a--packed year for Rare and and of Thieves, and as we position our sils towards that long- longed launch in early 2018, we’re also coming up on a full year of the Sea of Thists Teical Alp. It’s served the game incredibly well and given thousands upon thousands ofdding buccaneers a chance to hit the seas on Xbox One and Windows 10 PC – and we’ll soon be waving good to to the technical Al, in its current form, so this is your final chance to get involved!\n\nWhy sign up now if you haven’t already? Everyone who joins the Insider Programme by the cut-off date of Dec. 1 will be invited to the next play session – our biggest and best yet – with the latest brace of brand new features to test. go, as promised, all eligible insers are now getting their chance to jump in and play Sea of Th is before its release next year.\n\nIt was back in December 2016 that we announced the very first ical Al sessions session, following the kick-off of our Insider programme in November. In the space of a year we’ve scaled up from that first wave of 1,000 players to almost 200,000. We’ve debuted new features fromalletal sentinels to alone ships, welcomed your feedback and used it to balance and finetune all manner of things, honoring our promise to shape Sea of Thieves alongside its community. Now we’re preparing to welcome all those insiders still waiting to take to the waves, plus all new Insiders who sign up between now and Dec. 1.\n\nSo if you haven’t yet pounced on the possibility of playing playing of Thieves early, waste no more time and join the insider staff at the.seaofthieves.com/insider. Our Insiders receive regular updates from the development team through official, posts and newsletters, and of course get the chance to play the game and share their with with us as it continues to evolutionve.\n\nJoining the ins1 Programme is free – you just need to have an Xbox Live account (or set up"
# text2 = "2017 has been a jam-packed year for Rare and Sea of Thieves, and as we angle our sails towards that long-awaited launch in early 2018, we’re also coming up on a full year of the Sea of Thieves Technical Alpha. It’s served the game incredibly well and given thousands upon thousands of budding buccaneers a chance to hit the seas on Xbox One and Windows 10 PC – and we’ll soon be waving goodbye to the Technical Alpha in its current form, so this is your final chance to get involved!\n\nWhy sign up now if you haven’t already? Everyone who joins the Insider Programme by the cut-off date of Dec. 1 will be invited to the next play session – our biggest and best yet – with the latest brace of brand new features to test. Yes, as promised, all eligible Insiders are now getting their chance to jump in and play Sea of Thieves before its release next year.\n\nIt was back in December 2016 that we announced the very first Technical Alpha sessions, following the kick-off of our Insider Programme in November. In the space of a year we’ve scaled up from that first wave of 1,000 players to almost 200,000. We’ve debuted new features from skeletal sentinels to solo ships, welcomed your feedback and used it to balance and finetune all manner of things, honoring our promise to shape Sea of Thieves alongside its community. Now we’re preparing to welcome all those Insiders still waiting to take to the waves, plus all new Insiders who sign up between now and Dec. 1.\n\nSo if you haven’t yet pounced on the possibility of playing Sea of Thieves early, waste no more time and join the Insider crew at www.seaofthieves.com/insider. Our Insiders receive regular updates from the development team through official forum posts and newsletters, and of course get the chance to play the game and share their thoughts with us as it continues to evolve.\n\nJoining the Insider Programme is free – you just need to have an Xbox Live account (or set up a new one) and be 18 years or older to sign the NDA. As Sea of Thieves is an online multiplayer game, Xbox Live Gold membership is also required when sailing the seas on Xbox One. Please remember that these Technical Alpha play sessions are purely for the benefit of our Insiders and are currently under NDA, so they can’t be streamed or captured!\n\nFor more information about the Insider Programme including some details on eligibility, see our FAQ. And stay tuned to official Sea of Thieves and Xbox social channels for an announcement of the big play session date coming up in December.\n\nSee you all on the seas!\n\n"
# # 对文本进行分词并计算token的数量
# tokens1 = torch.tensor(tokenizer(text1)["input_ids"]).squeeze(0)[:510]
# tokens2 = torch.tensor(tokenizer(text2)["input_ids"]).squeeze(0)[:510]
# with torch.no_grad():
#     embeddings1 = model.model.embed_tokens(tokens1)  # 计算文本1的平均嵌入
#     embeddings2 = model.model.embed_tokens(tokens2) # 计算文本2的平均嵌入
#     correct_tokens += sum(1 for o,d in zip(tokens1, tokens2) if o == d)
# # acc = cosine_similarity(embeddings1,embeddings2).mean()
# print("embeddings1:", embeddings1)
# print("embeddings2:", embeddings2)
# # 计算两个文本平均嵌入的余弦相似度
# # cos_sim = cosine_similarity(embeddings1,embeddings2).mean()
# # print(cos_sim)
#
# # print("文本相似度:", cos_sim)
# print("acc2:", correct_tokens / len(tokens2))
# 11
# text1 = "An investment made by Kobe Bryant has yielded more than 30 times its money in fewer than four and a half years.\n\nOn Tuesday, Coca-Cola announced it had purchased a minority stake in sports drink BodyArmor.\n\nBryant made his first investment in the brand, for roughly 10 percent of the company, in March 2014, putting in roughly $6 million over time. Based on the valuation of the Coca-Cola deal, his stake is now worth approximately $200 million, sources told ESPN.\n\nBryant is now the fourth-largest investor in the brand, marketed as a healthier competitor to Gatorade, behind the brand's co-founder Mike Repole, Coca-Cola and Keurig Dr Pepper. When Bryant invested in BodyArmor, the brand had just come off a year of $10 million in sales. BodyArmor is projected to top $400 million in sales in 2018.\n\nKobe Bryant's other March 2014 investment yielded an Oscar for \"Dear Basketball.\" Kevin Winter/Getty Images\n\nBryant, who earned $328 million on the court in his 20-year NBA career and a similar amount off the court over that time, announced his investment in BodyArmor on the same day he announced the start of his new company, Kobe Inc. He since has formed a $100 million joint venture investment firm with entrepreneur Jeff Stibel and started his own production company, Granity Studios, which won an Oscar in 2018 for best animated short for his \"Dear Basketball\" film.\n\nAs part of their endorsement deals, many athletes had equity stakes in BodyArmor. Sources told ESPN that as many as a dozen superstar athletes could also have stakes in BodyArmor worth more than $1 million, including James Harden, Dustin Johnson and Andrew Luck.\n\nCoca-Cola's acquisition is the biggest story in the business of sports drinks since December 2000, when PepsiCo acquired Quaker Oats, which included Gatorade. The deal puts BodyArmor in Coke's powerful distribution network, on their delivery trucks throughout most of the United States.\n\nThis is the second time Repole has sold a company to Coca-Cola. In 2007, Glaceau, a company he co-founded with the smartwater and vitaminwater brands, sold to Coke for $4.1 billion.\n\nBryant's return is the biggest return for a modern-day athlete in the business world in some time. LeBron James made $30 million from a small stake in Beats by Dre when it sold to Apple in May 2014. James and his business partner Maverick Carter put less than $1 million into fast-casual pizza chain Blaze in 2012. That investment is now worth approximately $40 million.\n\n"
# text2 = "An investment made by K Bry Bryant has hased more than 30 times its money in fewer than four and a half years.\n\nOn Tuesday, Coca-Cola announced it had purchased a minority stake in sports drink.Ar..\n\nBryant made his first investment in the brand, for roughly 10 percent of the company, in March 2014, putting in roughly $6 million over time. based on the valueation of the Coca-Cola deal, his stake is now worth approximately $200 million, sources told ESPN.\n\nBryant is now the fourth- largestest investor in the brand, marketed as a healthier competitor to Gatorade, behind the brand's co-founder Mike Re Re, Coca-Cola and Keurig Dr Pepper. Whenryant invested in\nAr brand, the brand had just come off a year of $10 million in sales.\nAr is is projected to top $400 million in sales in 2018.\n\nKBanant's other March 2014 investment.ed an Oscar for \"Dear basketball.\" Kevin Kevin/Getty Images\n\nBryant, who earned $328 million on the court in his 20-year NBA career and a similar amount off the court over that time, announced his investment in bodyAror on the same day he announced the start of his new company, K. Inc. He since has formed a $100 million joint venture investment firm with entrepreneur Jeff Stibel and started his own production company, Grandity Studios, which won an Academy in 2018 for best animated short for his \"Dear basketball\" film.\n\nAs part of their endorsement deals, many athletes had equity stakes in bodyArAr. Sources told ESPN that as many as a dozen super stars athletes could also have stakes in bodyArAr worth more than $1 million, including James Hard, Dustin Johnson and Andrew Luck.\n\nCoca-ocaa's acquisition is the biggest story in the business of sports drinks since December 2000, when Pe hisCo acquired Quaker Oats, which included Gatorade. The"
# 95
# text1 = "California State Assemblyman Reggie Jones-Sawyer (D-Paramount) kicked off Tuesday’s confirmation hearing for newly-appointed state Attorney General Xavier Becerra by predicting a “legal war” between California and the Trump administration.\n\nThe San Diego Union-Tribune and other outlets reported that Jones-Sawyer said California faced a “long, legal war” with President-elect Donald Trump after he ran “the most xenophobic campaign in modern history.”\n\n“The incoming president ran the most xenophobic campaign in modern history,” says Assemblyman @JonesSawyer59 in opening of Becerra A hearing — John Myers (@johnmyers) January 10, 2017\n\nBecerra, who is currently a U.S. congressman and prominent member of the Democratic caucus, was nominated to take over as California Attorney General by Gov. Jerry Brown after Kamala Harris won a U.S. Senate seat. She was sworn in last week. Thus far, he has built his candidacy for the job around a promise to resist the policies of the Trump administration, especially on immigration. Separately, the state legislature has retained former U.S. Attorney General Eric Holder for the same purpose — though the constitutionality of that hire is being challenged by Republican State Assemblyman Kevin Kiley (R-Roseville).\n\nBrown introduced Becerra at the State Assembly’s hearing by warning that “there are big battles ahead.” In his own remarks, Becerra voiced an apparently newfound interest in states’ rights, vowing to resist “federal intrusion” in California: “You will find me being as aggressive as possible working with all of you to figure out ways that we can make sure there is no federal intrusion in areas that are really left to the state in the U.S. Constitution,” he said, according to the Los Angeles Times.\n\nImmigration, as Democrats have repeatedly noted, is an area under the exclusive jurisdiction of the federal government.\n\nThe State Assembly panel voted 6-3 on party lines to recommend Becerra’s appointment, the Times reported.\n\nJoel B. Pollak is Senior Editor-at-Large at Breitbart News. He was named one of the “most influential” people in news media in 2016. His new book, How Trump Won: The Inside Story of a Revolution, is available from Regnery. Follow him on Twitter at @joelpollak.\n\n",
# text2 = "California State Assemblyman Reg people Jones-Sawyer (D-),ount) kicked off Tuesday’s confirmation hearing for new-appointed state Attorney General X X Becerra by predicting a “ legal war” between California and the Trump administration.\n\nThe San Diego Union-Tribune and other outlets reported that Jones-Saw has said California faced a “long, legal war” with President-elect Donald Trump after he ran “the most xenophobic campaign in modern history.”\n\n“TheThe president ran the most xenophobic campaign in modern history,” says Assemblyman @JonesSawyer59 in opening of Becerra A hearing — John Myers (@j onemyers) January 10, 2017\n\nBecerra, who is currently a U.S. congressman and prominent member of the Democratic caucus, was nominated to take over as California Attorney General by Gov. Jerry Brown after Kamala Harris won a U.S. Senate seat. She was sworn in last week. thus far, he has built his hisacy for the job around a promise to to the policies of the Trump administration, especially on immigration. Separately, the state legislature has retain former U.S. Attorney Generalic Hale for the same purpose — though the constitution that of that hire is being challenged by Republican State Assemblyman Kevin Kiley (R-Roseville).\n\nBrown introduced Becerra at the State Assembly’s hearing by warning that “there are big battles ahead.” In his own remarks, Becerra areiced an apparently newfound interest in states’ rights, vowing to resist “fFal intrusion” in California: “You will find me being as aggressive as possible working with all of you to figure out ways that we can make sure there is no federal intrusion in areas that are really left to the state in the U.S. Constitution,” he said, according to the Los Angeles Times.\n\nImmigration, as Democrats have noted noted, is an area under the exclusive exclusive juris of the federal government.\n\nThe State Assembly panel voted 6-3 on party lines to recommend Becerra’s appointment, the Times reported.\n\n Joel B. Polur is senior editor"
# text = "Hello, this is an example string"
# marker = "an "
#
# # 使用split，去掉"example"及其之前的内容
# result = text.split(marker, 1)[-1]
# print(result)  # 输出 " string"

# count = 0
# with open(f'../train_instruction_dataset.json', 'r', encoding='utf-8') as f:
#     examples_list = json.load(f)
#     for example in tqdm(examples_list, desc="Processing examples"):
#         count += 1
#         if count < 10000:
#             continue
#         else:
#             print(example)



# from datasets import load_dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from torch import nn
# import os
# import random
# from tqdm import tqdm
# import argparse
# import json
#
#
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--work_dir', type=str, default="compressLLM_pwc_ntp", required=False,
#                         help='Directory including the configuration file')
#     return parser.parse_args()
#
#
# def get_long_text_list(dataset_repo):
#     # cache long text for preventing full dataset traversal on each preparation.
#     if os.path.exists('pwc_long_text.json'):
#         with open('pwc_long_text.json', 'r', encoding='utf-8') as f:
#             long_text_list = json.load(f)
#         return long_text_list
#
#     dataset = load_dataset(dataset_repo, split="train", streaming=True)
#
#     long_text_list = []
#     for example in tqdm(dataset, desc="Processing examples"):
#         if len(example["input"]) >= 600:
#             long_text_list.append(example["input"])
#
#     with open('pwc_long_text.json', 'w', encoding='utf-8') as f:
#         json.dump(long_text_list, f, ensure_ascii=False)
#
#     return long_text_list
#
#
# def get_examples(model_id, dataset_repo="DKYoon/SlimPajama-6B", hf_token=None, token_num=1_000_000_000, min_len=512,
#                  instruction_dataset_repo=None):
#     model_name = model_id.split('/')[-1]
#     train_data_name = "pwc_train_" + model_name + "_" + str(token_num) + f"token_len-{min_len}.pt"
#     eval_data_name = "pwc_eval_" + model_name + "_" + str(token_num) + f"token_len-{min_len}.pt"
#     print(f"in:train_data_name:{train_data_name}")
#     if os.path.exists(train_data_name):
#         print("loading data...")
#         return torch.load(train_data_name), torch.load(eval_data_name)
#     print(f"preparing data :train_data_name:{train_data_name}")
#
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype=torch.bfloat16,
#         device_map="cpu",
#         token=hf_token
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
#
#     long_text_list = get_long_text_list(dataset_repo)
#
#     examples = []
#     for text in tqdm(long_text_list, desc="Processing examples"):
#
#         ids = tokenizer(text)["input_ids"]
#
#         if len(ids) < min_len * 2:
#             continue
#
#         # half for prefix, half for LM
#         last_start = len(ids) - min_len * 2
#         random_start = random.randint(0, last_start)
#
#         # inputs = torch.LongTensor(ids[random_start:random_start+min_len])
#         # lm_target = torch.LongTensor(ids[random_start+min_len:random_start+2*min_len])
#         inputs = torch.LongTensor(ids[0:510])
#         lm_target = torch.LongTensor(ids[510:len(ids)])
#         examples.append({"inputs": inputs, "lm_target": lm_target})
#
#         if len(examples) * min_len >= token_num:
#             break
#
#     # 1k for validation
#     torch.save(examples[1000:], train_data_name)
#     torch.save(examples[:1000], eval_data_name)
#
#     return examples[1000:], examples[:1000]
#
#
# if __name__ == "__main__":
#     args = parse_args()
#     with open(args.work_dir + "/config.json") as f:
#         config = json.load(f)
#
#     training_config = config["training_config"]
#     config["data_config"]["model_id"] = training_config["model_id"]
#
#     # print(config["data_config"])
#     train_examples, eval_examples = get_examples(**config["data_config"])
#     # print(len(train_examples))
#     # print(train_examples[50])
#
# """
# python pre_prepare_data.py --work_dir CompressLLM
#
# unset HF_HUB_OFFLINE
# HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0 python pre_prepare_data.py --work_dir compressLLM_len-510_ratio-15
# HF_ENDPOINT=https://hf-mirror.com python pre_prepare_data.py --work_dir compressLLM_len-510_ratio-15
# """

# work_dir = "../compressLLM_random_instruction_(pre-train-multi-lora)_multi-lora_lm&cl"
# with open(work_dir + f'/config.json') as f:
#     config =json.load(f)
#
# config["data_config"]["model_id"] = "../../../models/TinyLlama/TinyLlama_v1.1"
# world_size = torch.cuda.device_count()
# tokenizer = AutoTokenizer.from_pretrained(config["data_config"]["model_id"], token=config["data_config"]["hf_token"])
#
# print("calculate BLEU4...")
#
#
# reference = "Komal Uzair and her brother Shayan completed their summit on the ",
# candidate = "Komal Uzair and her brother Shayan completed their summit on apple",
#
# input_ids = tokenizer(reference, add_special_tokens=False)["input_ids"][0]
# cl_gen_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"][0]
# bleu4 = sentence_bleu([input_ids], cl_gen_ids, weights=(0.25, 0.25, 0.25, 0.25))
#
#
# print(f"BLEU-4 Score: {bleu4}")


# position_ids = torch.arange(1,5)
# print(position_ids)
# logit = torch.tensor([1,2,3,4,8,2,6])
# next_token_id = torch.argmax(logit).tolist()
# print(next_token_id)

# with torch.no_grad():
#     model.eval()
#     teacher_outputs = model(
#         input_ids=batch['input_ids'],
#         attention_mask=batch['attention_mask'],
#     )
# def get_kl_loss(teacher_logits, student_logits, student_labels, teacher_labels, temperature, distill_topk=None):
#     ## make sure the teacher_logits and student_logits have the same shape
#     loss_fct = nn.KLDivLoss(reduction="batchmean")
#     _, _, vocab_size = student_logits.shape
#
#     ## only compute loss in the completion part, not propmt
#
#     student_mask = (student_labels != -100).unsqueeze(-1).expand_as(student_logits)  ## batch_size,num_tokens,vocab_size
#     student_logits_selected = torch.masked_select(student_logits, student_mask).view(-1, vocab_size)
#
#     teacher_mask = (teacher_labels != -100).unsqueeze(-1).expand_as(teacher_logits)
#     teacher_logits_selected = torch.masked_select(teacher_logits, teacher_mask).view(-1, vocab_size)
#
#     if distill_topk is not None:
#         _, topk_teacher_indices = torch.topk(teacher_logits_selected, k=distill_topk, dim=-1)
#
#         teacher_logits_selected = torch.gather(teacher_logits_selected, 1, topk_teacher_indices)
#         student_logits_selected = torch.gather(student_logits_selected, 1, topk_teacher_indices)
#
#     assert teacher_logits_selected.shape == student_logits_selected.shape, (
#         f"The shape of teacher logits is {teacher_logits_selected.shape}, while that of student is {student_logits_selected.shape}")
#
#     kl_loss = loss_fct(
#         F.log_softmax(student_logits_selected / temperature, dim=-1),
#         F.softmax(teacher_logits_selected / temperature, dim=-1),
#     ) * temperature ** 2
#
#     return kl_loss
#
# def prepare_inputs_embeds(self, input_position_ids, compress_position_ids, inputs_token, compress_token):
#
#     inputs_token[input_position_ids == compress_position_ids] = compress_token
#     inputs_token[input_position_ids != compress_position_ids] = -1
#
#     return inputs_token




# import torch
#
# # 示例数据
# inputs_token = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).float()
# input_position_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#
# compress_token = torch.tensor([101, 102, 103, 104]).float()
# compress_position_ids = torch.tensor([2, 4, 6, 8])
#
# # 找到 compress_position_ids 在 input_position_ids 中的索引
# indices = torch.isin(input_position_ids, compress_position_ids).nonzero(as_tuple=True)[0]
#
# # 替换对应位置
# inputs_token[indices] = compress_token
#
# print(inputs_token)

import logging
import pdb
import queue

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import math
import transformers

from compress.modify_code import modify_llama


class TripleLinearLoraLayer(nn.Module):
    def __init__(self, in_features, out_features, r_cl=16, r_lm=16, r_cl_prime=16, scale=1.0, weight=None):
        super(TripleLinearLoraLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = 2  # Scaling factor

        # 原始权重矩阵 W
        self.weight = nn.Parameter(weight, requires_grad=False)

        # 压缩 token 的 LoRA 模块参数 (A_cl, B_cl)
        self.lora_A_cl = nn.Parameter(torch.zeros((in_features, r_cl), device=self.weight.device, dtype=torch.bfloat16),
                                      requires_grad=False)
        self.lora_B_cl = nn.Parameter(
            torch.zeros((r_cl, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=False)

        # LLM token 的 LoRA 模块参数 (A_lm, B_lm)
        self.lora_A_lm = nn.Parameter(torch.zeros((in_features, r_lm), device=self.weight.device, dtype=torch.bfloat16),
                                      requires_grad=True)
        self.lora_B_lm = nn.Parameter(
            torch.zeros((r_lm, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

        # 额外的压缩 token 的 LoRA 模块参数 (A_cl', B_cl')
        self.lora_A_cl_prime = nn.Parameter(
            torch.zeros((in_features, r_cl_prime), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_cl_prime = nn.Parameter(
            torch.zeros((r_cl_prime, out_features), device=self.weight.device, dtype=torch.bfloat16),
            requires_grad=True)

        # 初始化 LoRA 参数
        nn.init.kaiming_uniform_(self.lora_A_cl, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_cl)
        nn.init.kaiming_uniform_(self.lora_A_lm, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_lm)
        nn.init.kaiming_uniform_(self.lora_A_cl_prime, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_cl_prime)

    def forward(self, x, mask):
        # 原始权重的计算结果，只计算一次
        result = F.linear(x, self.weight)

        # 检查并应用每种 mask
        if "cl_mask" in mask:
            x_cl = x * mask["cl_mask"]
            result_cl = self.scale * (x_cl @ self.lora_A_cl @ self.lora_B_cl)
            result += result_cl

        if "lm_mask" in mask:
            x_lm = x * mask["lm_mask"]
            result_lm = self.scale * (x_lm @ self.lora_A_lm @ self.lora_B_lm)
            result += result_lm

        if "cl_prime_mask" in mask:
            x_cl_prime = x * mask["cl_prime_mask"]
            result_cl_prime = self.scale * (x_cl_prime @ self.lora_A_cl_prime @ self.lora_B_cl_prime)
            result += result_cl_prime

        return result


class TripleEmbeddingLoraLayer(nn.Module):
    def __init__(self, in_features, out_features, padding_idx, r_cl=128, r_lm=128, r_cl_prime=128, scale=1.0,
                 weight=None):
        super(TripleEmbeddingLoraLayer, self).__init__()
        self.num_embeddings = in_features
        self.embedding_dim = out_features
        self.padding_idx = padding_idx
        self.scale = 2  # Scaling factor

        # 原始权重矩阵 W
        self.weight = nn.Parameter(weight, requires_grad=False)

        # 压缩 token 的 LoRA 模块参数 (A_cl, B_cl)
        self.lora_A_cl = nn.Parameter(torch.zeros((in_features, r_cl), device=self.weight.device, dtype=torch.bfloat16),
                                      requires_grad=False)
        self.lora_B_cl = nn.Parameter(
            torch.zeros((r_cl, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=False)

        # LLM token 的 LoRA 模块参数 (A_lm, B_lm)
        self.lora_A_lm = nn.Parameter(torch.zeros((in_features, r_lm), device=self.weight.device, dtype=torch.bfloat16),
                                      requires_grad=True)
        self.lora_B_lm = nn.Parameter(
            torch.zeros((r_lm, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

        # 额外的压缩 token 的 LoRA 模块参数 (A_cl', B_cl')
        self.lora_A_cl_prime = nn.Parameter(
            torch.zeros((in_features, r_cl_prime), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_cl_prime = nn.Parameter(
            torch.zeros((r_cl_prime, out_features), device=self.weight.device, dtype=torch.bfloat16),
            requires_grad=True)

        # 初始化 LoRA 参数
        nn.init.zeros_(self.lora_A_cl)
        nn.init.normal_(self.lora_B_cl)
        nn.init.zeros_(self.lora_A_lm)
        nn.init.normal_(self.lora_B_lm)
        nn.init.zeros_(self.lora_A_cl_prime)
        nn.init.normal_(self.lora_B_cl_prime)

    def forward(self, x, mask):
        # 计算一次嵌入的基准结果
        result = F.embedding(x, self.weight, self.padding_idx)  # 初始化结果

        # 检查每个 mask 并应用相应的 LoRA 层
        if "cl_mask" in mask:
            x_cl = x * mask["cl_mask"]
            after_A_cl = F.embedding(x_cl, self.lora_A_cl, self.padding_idx)
            result_cl = self.scale * (after_A_cl @ self.lora_B_cl)
            result += result_cl

        if "lm_mask" in mask:
            x_lm = x * mask["lm_mask"]
            after_A_lm = F.embedding(x_lm, self.lora_A_lm, self.padding_idx)
            result_lm = self.scale * (after_A_lm @ self.lora_B_lm)
            result += result_lm

        if "cl_prime_mask" in mask:
            x_cl_prime = x * mask["cl_prime_mask"]
            after_A_cl_prime = F.embedding(x_cl_prime, self.lora_A_cl_prime, self.padding_idx)
            result_cl_prime = self.scale * (after_A_cl_prime @ self.lora_B_cl_prime)
            result += result_cl_prime

        return result


# from peft import prepare_model_for_kbit_training

class LinearLoraLayer(nn.Module):
    # No bias in LLama3 LinearLayer
    def __init__(self, in_features, out_features, r=16, weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = 2  # The alpha value is usually twice the rank
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16),
                                   requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16),
                                   requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = F.linear(x, self.weight)
        result += self.scale * (x @ self.lora_A @ self.lora_B)
        return result


class EmbeddingLoraLayer(nn.Module):
    # No bias in LLama3 LinearLayer
    def __init__(self, in_features, out_features, padding_idx, r=128, weight=None):
        super().__init__()
        self.num_embeddings = in_features
        self.embedding_dim = out_features
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = 2  # The alpha value is usually twice the rank
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16),
                                   requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16),
                                   requires_grad=True)
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B)

    def forward(self, x):
        result = F.embedding(x, self.weight, self.padding_idx)
        after_A = F.embedding(x, self.lora_A, self.padding_idx)
        result += self.scale * (after_A @ self.lora_B)
        return result


class CompressLLM(torch.nn.Module):
    def __init__(self, model_id, mem_size, head_num, device_rank, task_config):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{device_rank}",
        )
        self.device = f"cuda:{device_rank}"
        self.task_config = task_config
        config = self.model.config
        self.vocab_size = config.vocab_size
        self.mem_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((mem_size, config.hidden_size)),
                                       requires_grad=True)
        self.special_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((2, config.hidden_size)),
                                           requires_grad=True)
        self.head_num = head_num

        self.compress_head = nn.Linear(config.hidden_size, head_num * config.vocab_size, bias=False,
                                       device=f"cuda:{device_rank}",
                                       dtype=self.model.model.embed_tokens.weight.dtype)

        # self.compress_head = nn.Sequential(
        #     nn.Linear(config.hidden_size, head_num*128, bias=False, device=f"cuda:{device_rank}", dtype=self.model.model.embed_tokens.weight.dtype),
        #     nn.Linear(head_num*128, head_num*config.vocab_size, bias=False, device=f"cuda:{device_rank}", dtype=self.model.model.embed_tokens.weight.dtype)
        #     )
        mean = torch.mean(self.model.model.embed_tokens.weight).item()
        std = torch.std(self.model.model.embed_tokens.weight).item()
        nn.init.normal_(self.mem_tokens, mean=mean, std=std)
        nn.init.normal_(self.special_tokens, mean=mean, std=std)

    def forward(self, inputs):
        # ->LlamaForCausalLM->LlamaModel->embed_tokens
        if self.task_config["use_multi_lora"]:
            mask = {"lm_mask": torch.ones_like(inputs['input_ids'])}
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"], mask)
        else:
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"])
        bsz, seq_len, emb_size = inputs_embeds.size()
        mem_size = self.mem_tokens.size(0)
        expand_mem = self.mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
        encode_inputs_embeds = torch.cat([inputs_embeds, expand_mem], dim=1)

        # [1,seq_len]
        position_ids = torch.arange(1, seq_len + 1, device=inputs_embeds.device).unsqueeze(0)
        # [1,mem_size]
        mem_position_ids = torch.arange((self.head_num + 1) // 2, self.head_num * mem_size + 1, step=self.head_num,
                                        device=inputs_embeds.device).unsqueeze(0)
        # [1,seq_len+mem_size]
        encode_position_ids = torch.cat([position_ids, mem_position_ids], dim=1)

        # print(f"encode_inputs_embeds:{encode_inputs_embeds.shape}")
        # print(f"position_ids:{position_ids.shape}, mem_position_ids:{mem_position_ids.shape}")

        # make three masks：cl_mask、lm_mask、cl_prime_mask
        if self.task_config["use_multi_lora"]:
            mask = make_masks(inputs_embeds, expand_mem)

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            if "wo_pe" in self.task_config:
                # print("here no pe")
                outputs = self.model(
                    inputs_embeds=encode_inputs_embeds,
                    output_hidden_states=True,
                    mask=mask,
                )
            else:
                outputs = self.model(
                    position_ids=encode_position_ids,
                    inputs_embeds=encode_inputs_embeds,
                    output_hidden_states=True,
                    mask=mask,
                )
        else:
            if "wo_pe" in self.task_config:
                outputs = self.model(
                    inputs_embeds=encode_inputs_embeds,
                    output_hidden_states=True,
                )
            else:
                outputs = self.model(
                    position_ids=encode_position_ids,
                    inputs_embeds=encode_inputs_embeds,
                    output_hidden_states=True,
                )

        hidden_states = outputs.hidden_states[-1]

        # [B,mem_size,emb_size]
        mem_hidden = hidden_states[:, -mem_size:]
        # [B,seq_len,vocab_size]
        original_logits = outputs.logits[:, :seq_len]

        tot_loss = 0
        tot_task = 0
        loss_info = {}

        use_cmp = False
        if "instruction_fine-tuning_add_compress_loss" in self.task_config and self.task_config[
            "instruction_fine-tuning_add_compress_loss"]:
            use_cmp = True
        # compress loss：压缩的是输入的context，并不是prompt和answer
        if use_cmp:
            # print("compress_targets will be used")
            # [B,mem_size,emb_size] -> [B,mem_size,head_num*vocab_size]
            logits = self.compress_head(mem_hidden)

            # extract original logits
            # [B,mem_size,head_num*vocab_size] -> [B,tot_Seg_len,V] -> [B,seq_len,V]
            logits = logits.reshape(bsz, mem_size * self.head_num, self.vocab_size)
            logits = logits[:, :seq_len, :]

            logits = logits.float()
            logits = logits.contiguous().view(-1, self.vocab_size)

            compress_targets = inputs["input_ids"].contiguous().view(-1).to(logits.device)

            compress_loss = self.loss_fct(logits, compress_targets)
            loss_info["compress_loss"] = compress_loss.item()
            tot_loss += compress_loss
            tot_task += 1

            # LM loss
        if 'lm_targets' in inputs and self.task_config["use_lm_loss"]:

            if inputs['lm_targets'] is None:
                if original_logits.shape[1] != inputs["instruction_target"].shape[
                    1]:  # if only <eos> in next segment, they will be equal.
                    # no token after <eos> [context + prompt + answer 510]
                    original_logits = original_logits[:, :-1]
                logits = original_logits.contiguous().view(-1, self.vocab_size)
                inputs["instruction_target"] = inputs["instruction_target"].contiguous().view(-1).to(logits.device)

                lm_loss = self.loss_fct(logits, inputs["instruction_target"])
                loss_info["lm_loss"] = lm_loss.item()
                return {"loss": lm_loss, "loss_info": loss_info}

            if self.task_config["use_multi_lora"]:
                mask = {"lm_mask": torch.ones_like(inputs['lm_targets'][:, :-1])}
                # [B,seq_len-1] -> [B,seq_len-1,E]
                lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'][:, :-1], mask)
            else:
                lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'][:, :-1])

            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)

            # todo: 1.将mem_hidden设置为0, .detach()
            #  [B,mem_size,E];     [B,1,E];      [B,seq_len-1,E]
            lm_emb = torch.cat([mem_hidden, expand_lm_token, lm_target_emb], dim=1)

            latter_position_ids = torch.arange(seq_len, seq_len + 1 + lm_target_emb.size(1),
                                               device=inputs_embeds.device).unsqueeze(0)
            lm_position_ids = torch.cat([mem_position_ids, latter_position_ids], dim=1)

            # make three masks
            if self.task_config["use_multi_lora"]:
                mask = make_masks(torch.cat([expand_lm_token, lm_target_emb], dim=1), mem_hidden,
                                  compress_prime_token=True)
                if "wo_pe" in self.task_config:
                    outputs = self.model(
                        inputs_embeds=lm_emb,
                        mask=mask,
                    )
                else:
                    outputs = self.model(
                        position_ids=lm_position_ids,
                        inputs_embeds=lm_emb,
                        mask=mask,
                    )
            else:
                if "wo_pe" in self.task_config:
                    outputs = self.model(
                        inputs_embeds=lm_emb,
                    )
                else:
                    outputs = self.model(
                        position_ids=lm_position_ids,
                        inputs_embeds=lm_emb,
                    )

            # [B,mem_size+S,V] -> [B,S,V]
            logits = outputs.logits[:, mem_size:]

            # here, we cat the whole seq's logits
            logits = torch.cat([original_logits, logits[:, 1:]], dim=1)
            logits = logits.contiguous().view(-1, self.vocab_size)
            inputs["instruction_target"] = inputs["instruction_target"].contiguous().view(-1).to(logits.device)

            lm_loss = self.loss_fct(logits, inputs["instruction_target"])
            loss_info["lm_loss"] = lm_loss.item()
            tot_loss += lm_loss
            tot_task += 1

        loss = tot_loss / tot_task

        return {"loss": loss, "loss_info": loss_info}

    def lm_inference(self, inputs, segment_size):
        # ->LlamaForCausalLM->LlamaModel->embed_tokens
        if self.task_config["use_multi_lora"]:
            mask = {"lm_mask": torch.ones_like(inputs["input_ids"])}
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"], mask)
        else:
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"])
        bsz, seq_len, emb_size = inputs_embeds.size()
        mem_size = self.mem_tokens.size(0)

        # [1,seq_len]
        position_ids = torch.arange(1, seq_len + 1, device=inputs_embeds.device).unsqueeze(0)

        if inputs['lm_targets'] is None:
            generate_text = []
            past_key_values = None
            next_inputs_embeds = inputs_embeds.clone()
            next_position_ids = position_ids.clone()

            for i in range(4096):
                if self.task_config["use_multi_lora"]:
                    mask = {"lm_mask": torch.ones_like(next_inputs_embeds)}
                    if "wo_pe" in self.task_config:
                        out = self.model(inputs_embeds=next_inputs_embeds, past_key_values=past_key_values,
                                         use_cache=True, mask=mask)
                    else:
                        out = self.model(position_ids=next_position_ids, inputs_embeds=next_inputs_embeds,
                                         past_key_values=past_key_values, use_cache=True, mask=mask)
                else:
                    if "wo_pe" in self.task_config:
                        out = self.model(inputs_embeds=next_inputs_embeds, past_key_values=past_key_values,
                                         use_cache=True)
                    else:
                        out = self.model(position_ids=next_position_ids, inputs_embeds=next_inputs_embeds,
                                         past_key_values=past_key_values, use_cache=True)
                # [B,S,V] -> [B,V]
                logit = out.logits[:, -1]
                past_key_values = out.past_key_values
                # [B,V]->[B]
                next_token_id = torch.argmax(logit, dim=-1)

                # [B]->[B,E]->[B,1,E]
                if self.task_config["use_multi_lora"]:
                    mask = {"lm_mask": torch.ones_like(next_token_id)}
                    next_inputs_embeds = self.model.model.embed_tokens(next_token_id, mask).unsqueeze(1).to(
                        inputs_embeds.device)
                else:
                    next_inputs_embeds = self.model.model.embed_tokens(next_token_id).unsqueeze(1).to(
                        inputs_embeds.device)
                next_position_ids = next_position_ids[:, -1:] + 1  # [1, seq_len]/[1,1] -> [1,1]
                generate_text.append(next_token_id.item())
                if next_token_id.item() == 2:  # eos
                    return generate_text
                if next_position_ids.item() > segment_size:
                    expand_mem = self.mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
                    encode_inputs_embeds = expand_mem

                    # [1,mem_size]
                    mem_position_ids = torch.arange((self.head_num + 1) // 2, segment_size + 1, step=self.head_num,
                                                    device=inputs_embeds.device).unsqueeze(0)
                    # [1,seq_len+mem_size]
                    encode_position_ids = torch.cat([position_ids, mem_position_ids], dim=1)

                    if self.task_config["use_multi_lora"]:
                        mask = {"cl_mask": torch.ones_like(encode_inputs_embeds)}
                        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
                        if "wo_pe" in self.task_config:
                            outputs = self.model(
                                inputs_embeds=encode_inputs_embeds,
                                past_key_values=past_key_values,
                                use_cache=True,
                                output_hidden_states=True,
                                mask=mask,
                            )
                        else:
                            outputs = self.model(
                                position_ids=mem_position_ids,
                                inputs_embeds=encode_inputs_embeds,
                                past_key_values=past_key_values,
                                use_cache=True,
                                output_hidden_states=True,
                                mask=mask,
                            )
                    else:
                        if "wo_pe" in self.task_config:
                            outputs = self.model(
                                inputs_embeds=encode_inputs_embeds,
                                past_key_values=past_key_values,
                                use_cache=True,
                                output_hidden_states=True,
                            )
                        else:
                            outputs = self.model(
                                position_ids=mem_position_ids,
                                inputs_embeds=encode_inputs_embeds,
                                past_key_values=past_key_values,
                                use_cache=True,
                                output_hidden_states=True,
                            )

                    hidden_states = outputs.hidden_states[-1]

                    # [B,mem_size,emb_size]
                    mem_hidden = hidden_states[:, -mem_size:]

                    # [1,E] -> [1,1,E] -> [B,1,E]
                    expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)

                    #                  [B,mem_size,E];     [B,1,E];
                    lm_emb = torch.cat([mem_hidden, expand_lm_token], dim=1)

                    #                              [1,mem_size];    [1,1];
                    lm_position_ids = torch.cat([mem_position_ids, next_position_ids - 1], dim=1)

                    past_key_values = None

                    if self.task_config["use_multi_lora"]:
                        mask = make_masks(mem_hidden, expand_lm_token, compress_prime_token=True)
                        if "wo_pe" in self.task_config:
                            out = self.model(inputs_embeds=lm_emb,
                                             past_key_values=past_key_values, use_cache=True, mask=mask)
                        else:
                            out = self.model(position_ids=lm_position_ids, inputs_embeds=lm_emb,
                                             past_key_values=past_key_values, use_cache=True, mask=mask)
                    else:
                        if "wo_pe" in self.task_config:
                            out = self.model(inputs_embeds=lm_emb,
                                             past_key_values=past_key_values, use_cache=True)
                        else:
                            out = self.model(position_ids=lm_position_ids, inputs_embeds=lm_emb,
                                             past_key_values=past_key_values, use_cache=True)
                    past_key_values = out.past_key_values

                    # next_token_id and next_position_ids don't be changed here.

        else:
            if self.task_config["use_multi_lora"]:
                mask = {"lm_mask": torch.ones_like(inputs['lm_targets'])}
                after_embeds = self.model.model.embed_tokens(inputs['lm_targets'], mask)
            else:
                after_embeds = self.model.model.embed_tokens(inputs['lm_targets'])
            expand_mem = self.mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
            encode_inputs_embeds = torch.cat([inputs_embeds, expand_mem], dim=1)

            # [1,mem_size]
            mem_position_ids = torch.arange((self.head_num + 1) // 2, segment_size + 1, step=self.head_num,
                                            device=inputs_embeds.device).unsqueeze(0)
            # [1,seq_len+mem_size]
            encode_position_ids = torch.cat([position_ids, mem_position_ids], dim=1)

            if self.task_config["use_multi_lora"]:
                mask = make_masks(inputs_embeds, expand_mem)
                # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
                if "wo_pe" in self.task_config:
                    outputs = self.model(
                        inputs_embeds=encode_inputs_embeds,
                        output_hidden_states=True,
                        mask=mask,
                    )
                else:
                    outputs = self.model(
                        position_ids=encode_position_ids,
                        inputs_embeds=encode_inputs_embeds,
                        output_hidden_states=True,
                        mask=mask,
                    )
            else:
                if "wo_pe" in self.task_config:
                    outputs = self.model(
                        inputs_embeds=encode_inputs_embeds,
                        output_hidden_states=True,
                    )
                else:
                    outputs = self.model(
                        position_ids=encode_position_ids,
                        inputs_embeds=encode_inputs_embeds,
                        output_hidden_states=True,
                    )

            hidden_states = outputs.hidden_states[-1]

            # [B,mem_size,emb_size]
            mem_hidden = hidden_states[:, -mem_size:]

            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)

            #                     [B,mem_size,E];     [B,1,E];      [B,seq_len-1,E]
            lm_emb = torch.cat([mem_hidden, expand_lm_token, after_embeds], dim=1)

            after_len = expand_lm_token.size(1) + after_embeds.size(1)
            after_position_ids = torch.arange(segment_size, segment_size + after_len,
                                              device=inputs_embeds.device).unsqueeze(0)
            #                              [1,mem_size];    [1,seq_len];
            lm_position_ids = torch.cat([mem_position_ids, after_position_ids], dim=1)


            generate_text = []
            past_key_values = None
            next_inputs_embeds = lm_emb.clone()
            next_position_ids = lm_position_ids.clone()
            if self.task_config["use_multi_lora"]:
                mask = make_masks(torch.cat([expand_lm_token, after_embeds], dim=1), mem_hidden,
                                  compress_prime_token=True)
            for i in range(100):
                # print(f"next_position_ids:{next_position_ids}")
                if self.task_config["use_multi_lora"]:
                    if "wo_pe" in self.task_config:
                        out = self.model(inputs_embeds=next_inputs_embeds,
                                         past_key_values=past_key_values,
                                         use_cache=True,
                                         mask=mask)
                    else:
                        out = self.model(position_ids=next_position_ids,
                                         inputs_embeds=next_inputs_embeds,
                                         past_key_values=past_key_values,
                                         use_cache=True,
                                         mask=mask)
                else:
                    if "wo_pe" in self.task_config:
                        out = self.model(inputs_embeds=next_inputs_embeds,
                                         past_key_values=past_key_values,
                                         use_cache=True)
                    else:
                        out = self.model(position_ids=next_position_ids,
                                         inputs_embeds=next_inputs_embeds,
                                         past_key_values=past_key_values,
                                         use_cache=True)
                # [B,S,V] -> [B,V]
                logit = out.logits[:, -1]
                past_key_values = out.past_key_values
                # [B,V]->[B]
                next_token_id = torch.argmax(logit, dim=-1)

                # [B]->[B,E]->[B,1,E]
                if self.task_config["use_multi_lora"]:
                    mask = {"lm_mask": torch.ones_like(next_token_id)}
                    next_inputs_embeds = self.model.model.embed_tokens(next_token_id, mask).unsqueeze(1).to(
                        inputs_embeds.device)
                    mask = {"lm_mask": torch.ones_like(next_inputs_embeds)}
                else:
                    next_inputs_embeds = self.model.model.embed_tokens(next_token_id).unsqueeze(1).to(
                        inputs_embeds.device)
                # todo: 不是很理解这里每次都是[1,1]和+1的作用
                next_position_ids = next_position_ids[:, -1:] + 1  # [1, seq_len]/[1,1] -> [1,1]
                generate_text.append(next_token_id.item())
                if next_token_id.item() == 2:
                    return generate_text

            return generate_text
        return generate_text

    def cl_inference(self, inputs, segment_size):
        # ->LlamaForCausalLM->LlamaModel->embed_tokens
        # todo:1.
        if self.task_config["use_multi_lora"]:
            mask = {"lm_mask": torch.ones_like(inputs['input_ids'])}
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"], mask)
        else:
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"])
        bsz, seq_len, emb_size = inputs_embeds.size()
        mem_size = self.mem_tokens.size(0)
        expand_mem = self.mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
        encode_inputs_embeds = torch.cat([inputs_embeds, expand_mem], dim=1)

        # [1,seq_len]
        position_ids = torch.arange(1, seq_len + 1, device=inputs_embeds.device).unsqueeze(0)
        # [1,mem_size]
        mem_position_ids = torch.arange((self.head_num + 1) // 2, segment_size + 1, step=self.head_num,
                                        device=inputs_embeds.device).unsqueeze(0)
        # [1,seq_len+mem_size]
        encode_position_ids = torch.cat([position_ids, mem_position_ids], dim=1)

        # todo:2.
        if self.task_config["use_multi_lora"]:
            mask = make_masks(inputs_embeds, expand_mem)
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            if "wo_pe" in self.task_config:
                # print("no pe in here")
                outputs = self.model(
                    inputs_embeds=encode_inputs_embeds,
                    output_hidden_states=True,
                    mask=mask,
                )
            else:
                outputs = self.model(
                    position_ids=encode_position_ids,
                    inputs_embeds=encode_inputs_embeds,
                    output_hidden_states=True,
                    mask=mask,
                )
        else:
            if "wo_pe" in self.task_config:
                # print("no pe in here")
                outputs = self.model(
                    inputs_embeds=encode_inputs_embeds,
                    output_hidden_states=True,
                )
            else:
                outputs = self.model(
                    position_ids=encode_position_ids,
                    inputs_embeds=encode_inputs_embeds,
                    output_hidden_states=True,
                )

        hidden_states = outputs.hidden_states[-1]

        # [B,mem_size,emb_size]
        mem_hidden = hidden_states[:, -mem_size:]
        # [B,mem_size,emb_size] -> [B,mem_size,head_num*vocab_size]
        logits = self.compress_head(mem_hidden).float()
        # [B*mem_size*head_num,vocab_size]
        logits = logits.contiguous().view(-1, self.vocab_size)
        # [b*m*h,v] -> [b*m*h]
        generate_text = torch.argmax(logits, dim=-1).tolist()

        return generate_text


def make_masks(input_token=None, compress_token=None, compress_prime_token=False):
    # make three masks：cl_mask、lm_mask、cl_prime_mask
    mask = {}
    if compress_prime_token:
        lm_zero_mask = torch.zeros_like(compress_token, dtype=torch.bfloat16).to(input_token.device)
        lm_ones_mask = torch.ones_like(input_token, dtype=torch.bfloat16).to(input_token.device)
        lm_mask = torch.cat([lm_zero_mask, lm_ones_mask], dim=1).to(input_token.device)

        cl_prime_ones_mask = torch.ones_like(compress_token, dtype=torch.bfloat16).to(input_token.device)
        cl_prime_zero_mask = torch.zeros_like(input_token, dtype=torch.bfloat16).to(input_token.device)
        cl_prime_mask = torch.cat([cl_prime_ones_mask, cl_prime_zero_mask], dim=1).to(input_token.device)

        mask.update({"cl_prime_mask": cl_prime_mask, "lm_mask": lm_mask, })
    else:
        cl_zero_mask = torch.zeros_like(input_token, dtype=torch.bfloat16).to(input_token.device)
        cl_ones_mask = torch.ones_like(compress_token, dtype=torch.bfloat16).to(input_token.device)
        cl_mask = torch.cat([cl_zero_mask, cl_ones_mask], dim=1).to(input_token.device)

        lm_ones_mask = torch.ones_like(input_token, dtype=torch.bfloat16).to(input_token.device)
        lm_zero_mask = torch.zeros_like(compress_token, dtype=torch.bfloat16).to(input_token.device)
        lm_mask = torch.cat([lm_ones_mask, lm_zero_mask], dim=1).to(input_token.device)

        mask.update({"cl_mask": cl_mask, "lm_mask": lm_mask, })
    return mask


def save_adapter(model, save_path_and_name='adapter.pt', log=False):
    adapter_name = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if log:
                print("[Save Adapter]", name)
            adapter_name.add(name)

    state_dict = model.state_dict()
    adapter_state_dict = {name: param for name, param in state_dict.items() if name in adapter_name}
    torch.save(adapter_state_dict, save_path_and_name)


def load_adapter(model, save_path_and_name='adapter.pt', log=False):
    adapter_state_dict = torch.load(save_path_and_name, map_location='cpu')  # 先加载到CPU
    if log:
        print("Loading adapter parameters:")
        for name, weight in adapter_state_dict.items():
            print(f"[Load Adapter] {name}")
    # 将adapter的权重转移到模型的设备上
    adapter_state_dict = {k: v.to(model.device) for k, v in adapter_state_dict.items()}

    model.load_state_dict(adapter_state_dict, strict=False)
    return model


def load_adapter_to_merge_weight(model, train_adapter='adapter.pt', instruction_adapter="", is_train=False):
    def merge_weight(model):
        for name, module in model.named_children():  # adapter是W'=W+AB -> instruction_adapter是
            if name == "compress_head":
                continue
            if isinstance(module, LinearLoraLayer) or isinstance(module, EmbeddingLoraLayer):
                lora_AB = module.lora_A.data @ module.lora_B.data
                if module.weight.data.shape == lora_AB.shape:
                    module.weight.data += lora_AB * module.scale
                else:
                    module.weight.data += lora_AB.transpose(0, 1) * module.scale
            else:
                merge_weight(module)

    def init_lora(model, task_config):
        for name, module in model.named_children():
            if name == "compress_head":
                continue
            if isinstance(module, LinearLoraLayer):
                setattr(model, name,
                        LinearLoraLayer(module.in_features, module.out_features, r=16,
                                        weight=module.weight.data.clone()))
            elif isinstance(module, EmbeddingLoraLayer):
                setattr(model, name,
                        EmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx, r=128,
                                           weight=module.weight.data.clone()))
            else:
                # Recursively apply this function to submodules
                init_lora(module, task_config)

    adapter_state_dict = torch.load(train_adapter, map_location='cpu')  # 先加载到CPU
    # 将adapter的权重转移到模型的设备上
    adapter_state_dict = {k: v.to(model.device) for k, v in adapter_state_dict.items()}

    model.load_state_dict(adapter_state_dict, strict=False)
    # W' -> W + AB
    merge_weight(model)
    init_lora(model, task_config="")
    # merge lora weight to origin
    if is_train:
        logging.info("train：merge lora weight to origin")
    else:
        # load A'B'
        adapter_state_dict = torch.load(instruction_adapter, map_location='cpu')  # 先加载到CPU
        # 将adapter的权重转移到模型的设备上
        adapter_state_dict = {k: v.to(model.device) for k, v in adapter_state_dict.items()}
        # finally -> h = W' + A'B' = W + AB + A'B'
        model.load_state_dict(adapter_state_dict, strict=False)
        logging.info("evaluator：no merge lora weight to origin")
    return model


def get_model_for_compress(model_id, task_config, rank):
    def add_compress_lora(model, task_config):
        for name, module in model.named_children():
            if name == "compress_head":
                continue
            if isinstance(module, nn.Linear):
                setattr(model, name,
                        LinearLoraLayer(module.in_features, module.out_features, r=16,
                                        weight=module.weight.data.clone()))
            elif isinstance(module, nn.Embedding):
                setattr(model, name,
                        EmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx, r=128,
                                           weight=module.weight.data.clone()))
            else:
                # Recursively apply this function to submodules
                add_compress_lora(module, task_config)

    def add_multi_lora(model, task_config):
        for name, module in model.named_children():
            if name == "compress_head":
                continue
            if isinstance(module, nn.Linear):
                setattr(model, name,
                        TripleLinearLoraLayer(module.in_features, module.out_features, r_cl=16, r_lm=16, r_cl_prime=16,
                                              weight=module.weight.data.clone()))
            elif isinstance(module, nn.Embedding):
                setattr(model, name,
                        TripleEmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx,
                                                 r_cl=128, r_lm=128, r_cl_prime=128, weight=module.weight.data.clone()))
            else:
                # Recursively apply this function to submodules
                add_multi_lora(module, task_config)

    # config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    if task_config["use_multi_lora"]:
        modify_llama()
        model = CompressLLM(
            model_id,
            mem_size=task_config["mem_size"],
            head_num=task_config["head_num"],
            device_rank=rank,
            task_config=task_config
        )
        add_multi_lora(model, task_config)
    else:
        model = CompressLLM(
            model_id,
            mem_size=task_config["mem_size"],
            head_num=task_config["head_num"],
            device_rank=rank,
            task_config=task_config
        )
        add_compress_lora(model, task_config)
    return model


def get_model(model_id, task_config, rank):
    if task_config["task_type"] == "Compress":
        return get_model_for_compress(model_id, task_config, rank)
    raise Exception("Don't exist [{task_type}] task.")


def load_model_with_adapter(model_id, task_config, rank, save_path_and_name='adapter.pt', log=False):
    model = get_model(model_id, task_config, rank)
    load_adapter(model, save_path_and_name, log)
    return model
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

# python /home/liuxinyu/zrs/forget-me-not/models/llama3.py

work_dir = "../compressLLM_random_instruction_(pre-train-multi-lora)_multi-lora_lm&cl"
with open(work_dir + f'/config.json') as f:
    config =json.load(f)

config["data_config"]["model_id"] = "../../../models/TinyLlama/TinyLlama_v1.1"
world_size = torch.cuda.device_count()
tokenizer = AutoTokenizer.from_pretrained(config["data_config"]["model_id"], token=config["data_config"]["hf_token"])

model = get_model(config["data_config"]["model_id"], config["task_config"], 0)
# model = load_adapter(model, save_path_and_name=work_dir+'/instruction_adapter.pt', log=False)
with torch.no_grad():
    model.eval()
    # 构造inputs{"input_ids:, lm_target:"}
    # context = "French senior civil servant arrested on suspicion of spying for North Korea\n\nNovember 27, 2018 by Joseph Fitsanakis\n\nA senior civil servant in the upper house of the French parliament has been arrested on suspicion of spying for North Korea, according to prosecutors. The news of the suspected spy’s arrest was first reported on Monday by Quotidien, a daily politics and culture show on the Monaco-based television channel TMC. The show cited “a judicial source in Paris” and said that France’s domestic security and counterintelligence agency, the General Directorate for Internal Security (DGSI), was in charge of the espionage case.\n\nThe senior administrator has been identified as Benoit Quennedey, a civil servant who liaises between the French Senate and the Department of Architecture and Heritage, which operates under France’s Ministry of Culture. Quennedey was reportedly detained on Sunday morning and his office in the French Senate was raided by DGSI officers on the same day. Quotidien said that he was arrested on suspicion of “collecting and delivering to a foreign power information likely to subvert core national interests”. The report did not provide specific information about the type of information that Quennedey is believed to have passed to North Korea. It did state, however, that a counterintelligence investigation into his activities began in March of this year.\n\nQuennedey is believed to be the president of the Franco-Korean Friendship Association, the French branch of a Spanish-based organization that lobbies in favor of international support for North Korea. Korea Friendship Association branches exist in over 30 countries and are believed to be officially sanctioned by Pyongyang. They operate as something akin to the pre-World War II Comintern (Communist International), a Moscow-sanctioned international pressure group that advocated in favor of Soviet-style communism around the world. French media reported on Monday that Quennedey traveled extensively to the Korean Peninsula in the past decade and has written a French-language book on North Korea. News reports said that the French President Emmanuel Macron had been made aware of Quennedey’s arrest. The senior civil servant faces up to 30 years in prison if found guilty of espionage.\n\n► Author: Joseph Fitsanakis | Date: 27 November 2018 | Permalink\n\n"
    # context = "Gone are the days when America’s standing in the world was contrasted primarily with that of the Soviet Union. Instead, the United States and China now compete to be the more favored world power.\n\nThe U.S. and China engender roughly the same level of goodwill. China is particularly well-liked in Latin America and the Middle East, while the U.S. fares better in Europe and the Asia-Pacific region.\n\nHowever, America’s weakening image in many nations has taken a toll on the country’s once-solid lead over China. And China’s own favorability has strengthened in recent years in Canada, Australia, Lebanon and Turkey.\n\nSince the most recent year Pew Research Center polled in 36 nations – 2014, 2015 or 2016, depending on the country – the number of nations in which the U.S. holds a competitive advantage in favorability over China has halved, from 25 to 12. (Differences of less than 6 percentage points are considered ties.) Whereas the U.S. once had a 12-point lead over China in terms of a global median, that lead has shrunk in 2017 to 2 points.\n\nIn six nations – Spain, Mexico, Turkey, Australia, Peru and Senegal – the dynamic between the two superpowers has flipped, with China overtaking the U.S. in favorability.\n\nAnd the United States’ once-significant lead over China in popularity has fallen to a virtual tie in another seven countries: Kenya, Germany, France, Brazil, Sweden, the UK and Canada.\n\nMeanwhile, in 12 nations, people view America more favorably than they do China: Vietnam, Israel, the Philippines, South Korea, Poland, Hungary, Italy, Ghana, Japan, South Africa, Colombia and India.\n\nA quarter-century after the collapse of the Soviet Union, Russia is viewed far less favorably than either the U.S. or China in most of the world, though America’s recent steep decline in image has improved Russia’s standing compared with that of the U.S.\n\nAmerica’s edge over Russia has contracted by more than 20 percentage points in 15 out of the 33 nations for which Pew Research Center has trend data on favorability toward Russia. These include Spain, France, Chile, Brazil, Italy, Australia and Tanzania.\n\nThe narrowing of the U.S.-Russia favorability gap is most striking in Mexico, where the 42-point advantage held by the U.S. over Russia in 2015 is all but gone. Mexicans now view the U.S. and Russia roughly the same.\n\n"
    # context = "Journey to the West is one of the four great classical novels of China. It was written by Wu Chengen, a writer in the Ming Dynasty. The novel takes the four Tang monks and their disciples as the main line, and integrates Taoism, Buddhism, Confucianism, as well as myths and legends, folk stories, historical stories and other elements in traditional Chinese culture, which has high literary value and historical status. The story is set in the Tang Dynasty, which mainly tells the story of four Tang monks, teachers and apprentices who went through ninety-eight one difficulties and finally obtained the true Scriptures. The following are the main characters and synopsis: 1. Tang Monk (Tang Sanzang) : Born Chen Xuanzang, a reincarnated son of the Golden cicada, he was ordered by Emperor Taizong to go to the West to learn scriptures. He was kind and compassionate, but sometimes too kind and gullible. 2. Monkey King (Monkey King) : Incarnated stone monkey, with 72 changes, somersault cloud and other supernatural powers. Smart, brave and loyal, he was a great disciple of the Tang Monk and was responsible for protecting his master's journey to the West. In the story, Sun Wukong saves Tang Monk and his disciples from danger many times. 3. Zhu Bajie (Wu Neng) : Originally a marshal of the sky Peng, he was banished to the world for molesting Chang 'e and was reborn as a pig. He was lazy, greedy for money and lustful, but honest and honest, was the Tang monk's two apprentices. 4. Sand Monk (Wujing) : Originally a rolling curtain general, he was relegated to the world for breaking glasses and became a sand monk. He was composed, loyal and reliable, and was one of the three disciples of the Tang Monk. The following is a summary of the story: 1. Tang Monk set out: Tang Monk was ordered by Emperor Taizong to go to the West Heaven to collect scriptures. Under the guidance of Guanyin Bodhisattva, he accepted three disciples, Sun Wukong, Zhu Bajie and Sha Seng successively. 2. Three dozen White Bone Essence: Tang Monk and his disciples encountered white bone essence through White Hu Ling. Sun Wukong discovers and defeats the White Bone Spirit three times, but Tang Monk misunderstands Wukong and drives him away. 3. Monkey King returns: The Tang Monk is captured by the Yellow Robe monster in the Treasure Elephant Country, and Zhu Bajie and Sha Seng are unable to rescue the master. After Sun Wukong learned the news, he returned to Master Men, surrendered the yellow robe monster, and rescued Tang Monk. 4. Red Boy: Tang Monk and his disciples pass through the Fire Cloud Cave and encounter Red Boy. Sun Wukong invited Guanyin Bodhisattva to surrender the Red Boy and make him a good fortune boy. 5. Daughter's Kingdom: Tang Monk's master and apprentice pass through the daughter's Kingdom, the king falls in love with Tang Monk and wants to recruit him as emperor's son-in-law. With the help of Sun Wukong, Tang Monk fled the Kingdom of Women. 6. True or false Monkey King: Sun Wukong killed the robber and was driven away by Tang Monk. Six-eared macaques pretend to be Sun Wukong and deceive Tang Monk. Finally, the Tathagata Buddha saw through the six-ear macaque, and Sun Wukong returned to Master. 7. 9981 difficulties: Tang Monk went through the 9981 difficulties and finally reached the West. With the help of the Goddess of Mercy Bodhisattva, the Tathagata Buddha and other divine Buddhas, we can obtain the true sutra. 8. Five holy achievements: The Tang monk and his disciples returned to the eastern soil and spread the true Sutra to the world. Tang Monk, Sun Wukong, Zhu Bajie, Sha Seng and Bai Longma were awarded the titles of Golden Arhat, Fighting Buddha, Pure Altar Messenger, Golden Arhat and eight Heavenly Dragon Guangzhi Bodhisattva respectively. Journey to the West, by telling the story of Tang Monk's masters and apprentices, conveys the values of loyalty, bravery, kindness and unity, which has high literary value and historical status."
    # context = "As of the census of 2000, there were 218,590 people, 79,667 households, and 60,387 families residing in the county. The population density was 496 people per square mile (192/km²). There were 83,146 housing units at an average density of 189 per square mile (73/km²). The racial makeup of the county was 86.77% Race (United States Census), 9.27% Race (United States Census), 0.23% Race (United States Census), 1.52% Race (United States Census), 0.06% Race (United States Census), 0.69% from Race (United States Census), and 1.47% from two or more races. 1.91% of the population were Race (United States Census) or Race (United States Census) of any race. 22.5% were of German people, 13.1% Irish people, 9.8% Italian people, 9.2% English, 8.1% American and 6.0% Polish ancestry."
    # context =  "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
    # context = "Controversial Michael Jackson Comedy Pulled by Sky Following Criticism From Family\n\n\"We set out to take a lighthearted look at reportedly true events and never intended to cause any offense,\" the company said about the episode of 'Urban Myths' featuring Joseph Fiennes as the late music star.\n\nU.K. pay-TV giant Sky said Friday it has decided not to air a TV program about Michael Jackson after his daughter, Paris Jackson, said she was \"incredibly offended\" by the portrayal of the late music star.\n\nThe episode was scheduled to be part of a series called Urban Myths and was set to air on Sky Arts on Jan. 19. It focused on Jackson's fabled road trip from New York to Los Angeles with Elizabeth Taylor and Marlon Brando after the 9/11 attacks.\n\nJoseph Fiennes played Jackson in a controversial decision, Stockard Channing portrayed Taylor, and Brian Cox played Brando.\n\n\"We have taken the decision not to broadcast Elizabeth, Michael and Marlon, a half-hour episode from the Sky Arts Urban Myths series, in light of the concerns expressed by Michael Jackson's immediate family,\" said Sky. \"We set out to take a lighthearted look at reportedly true events and never intended to cause any offense.\"\n\nSky added: \"Joseph Fiennes fully supports our decision.\"\n\nOn Thursday, Paris tweeted following the release of a trailer for the episode. She said the trailer made her \"want to vomit.\" She added: \"It angers me to see how obviously intentional it was for them to be this insulting, not just towards my father, but my godmother Liz as well.\"\n\nA petition to boycott the episode was launched and drew more than 20,000 signatures.\n\nAmid backlash last year to the casting of a white man as the King of Pop, Fiennes told The Hollywood Reporter that, though the program was \"not a biopic,\" he understood why people were \"up in arms.\"\n\n\"The decision with the casting and the producers — I wrangled with it, I was confused and shocked at what might come my way,\" said the actor. \"And I knew the sensitivity, especially to Michael's fans and to Michael's family. It doesn't negate who he was.\"\n\n"
    # context = "A static equilibrium between two forces is the most usual way of measuring forces, using simple devices such as weighing scales and spring balances. For example, an object suspended on a vertical spring scale experiences the force of gravity acting on the object balanced by a force applied by the \"spring reaction force\", which equals the object's weight. Using such tools, some quantitative force laws were discovered: that the force of gravity is proportional to volume for objects of constant density (widely exploited for millennia to define standard weights); Archimedes' principle for buoyancy; Archimedes' analysis of the lever; Boyle's law for gas pressure; and Hooke's law for springs. These were all formulated and experimentally verified before Isaac Newton expounded his Three Laws of Motion."
    # context = "Ctenophores may be abundant during the summer months in some coastal locations, but in other places they are uncommon and difficult to find. In bays where they occur in very high numbers, predation by ctenophores may control the populations of small zooplanktonic organisms such as copepods, which might otherwise wipe out the phytoplankton (planktonic plants), which are a vital part of marine food chains. One ctenophore, Mnemiopsis, has accidentally been introduced into the Black Sea, where it is blamed for causing fish stocks to collapse by eating both fish larvae and organisms that would otherwise have fed the fish. The situation was aggravated by other factors, such as over-fishing and long-term environmental changes that promoted the growth of the Mnemiopsis population. The later accidental introduction of Beroe helped to mitigate the problem, as Beroe preys on other ctenophores."
    # context = "The contracted batch of 15 Saturn Vs were enough for lunar landing missions through Apollo 20.  NASA publicized a preliminary list of eight more planned landing sites, with plans to increase the mass of the CSM and LM for the last five missions, along with the payload capacity of the Saturn V. These final missions would combine the I and J types in the 1967 list, allowing the CMP to operate a package of lunar orbital sensors and cameras while his companions were on the surface, and allowing them to stay on the Moon for over three days.  These missions would also carry the Lunar Roving Vehicle (LRV) increasing the exploration area and allowing televised liftoff of the LM.  Also, the Block II spacesuit was revised for the extended missions to allow greater flexibility and visibility for driving the LRV."
    # context = "Luther had been suffering from ill health for years, including Ménière's disease, vertigo, fainting, tinnitus, and a cataract in one eye. From 1531 to 1546, his health deteriorated further. The years of struggle with Rome, the antagonisms with and among his fellow reformers, and the scandal which ensued from the bigamy of the Philip of Hesse incident, in which Luther had played a leading role, all may have contributed. In 1536, he began to suffer from kidney and bladder stones, and arthritis, and an ear infection ruptured an ear drum. In December 1544, he began to feel the effects of angina."
    # context = "Conservative researchers have argued that income inequality is not significant because consumption, rather than income should be the measure of inequality, and inequality of consumption is less extreme than inequality of income in the US. Will Wilkinson of the libertarian Cato Institute states that \"the weight of the evidence shows that the run-up in consumption inequality has been considerably less dramatic than the rise in income inequality,\" and consumption is more important than income. According to Johnson, Smeeding, and Tory, consumption inequality was actually lower in 2001 than it was in 1986. The debate is summarized in \"The Hidden Prosperity of the Poor\" by journalist Thomas B. Edsall. Other studies have not found consumption inequality less dramatic than household income inequality, and the CBO's study found consumption data not \"adequately\" capturing \"consumption by high-income households\" as it does their income, though it did agree that household consumption numbers show more equal distribution than household income."
    # context = "Tesla was 6 feet 2 inches (1.88 m) tall and weighed 142 pounds (64 kg), with almost no weight variance from 1888 to about 1926. :292 He was an elegant, stylish figure in New York City, meticulous in his grooming, clothing, and regimented in his daily activities."

    # context = "Following the conquest of Dali in 1253, the former ruling Duan dynasty were appointed as governors-general, recognized as imperial officials by the Yuan, Ming, and Qing-era governments, principally in the province of Yunnan.  Succession for the Yuan dynasty, however, was an intractable problem, later causing much strife and internal struggle.  This emerged as early as the end of Kublai's reign.  Kublai originally named his eldest son, Zhenjin, as the Crown Prince, but he died before Kublai in 1285.  Thus, Zhenjin's third son, with the support of his mother Kökejin and the minister Bayan, succeeded the throne and ruled as Temür Khan, or Emperor Chengzong, from 1294 to 1307.  Temür Khan decided to maintain and continue much of the work begun by his grandfather.  He also made peace with the western Mongol khanates as well as neighboring countries such as Vietnam, which recognized his nominal suzerainty and paid tributes for a few decades.  However, the corruption in the Yuan dynasty began during the reign of Temür Khan"
    # -----------------文本题---------------------------------------------------------------------------------------------
    # context = "Hoping to break their current losing streak the Cowboys played on home ground for an Interconference duel with the Jaguars. In the first quarter the Cowboys took the lead as kicker David Buehler hit a 34-yard field goal. But they fell behind with QB David Garrard getting a 10-yard TD pass to WR Mike Sims-Walker. In the second quarter, the Cowboys struggled further with Garrard finding TE Marcedes Lewis on a 42-yard TD pass, then in the third quarter he found WR Mike Thomas on a 15-yard TD pass, and then he found Lewis again on a 9-yard TD pass. The Cowboys responded in the 4th quarter with RB Marion Barber getting a 1-yard TD run. But the Jaguars scored again with Garrard scrambling 2 yards to the endzone for a touchdown. The Cowboys replied with QB Jon Kitna making an 8-yard TD pass to TE Jason Witten."
    # question = "Which quarterback threw more touchdowns?"
    #
    # context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    # question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    # input_ids = tokenizer("### Context:\n")["input_ids"] + context_ids
    #
    # padding_token = tokenizer('\n', add_special_tokens=False)["input_ids"]
    # input_ids = pad_sequence(input_ids,510,padding_token)
    # lm_targets = tokenizer("\n### Question:\n", add_special_tokens=False)["input_ids"] + question_ids + tokenizer("\n### Answer:\n", add_special_tokens=False)["input_ids"]
    # inputs = {"input_ids":torch.tensor(input_ids).unsqueeze(0).to(model.device),
    #           "lm_targets":torch.tensor(lm_targets).unsqueeze(0).to(model.device)}
    # answer = model.lm_inference(inputs,segment_size=510)
    # answer = tokenizer.decode(answer, skip_special_tokens=True)
    # cl_generate_text = model.cl_inference(inputs, 510)
    # cl_generate_text = tokenizer.decode(cl_generate_text, skip_special_tokens=True)
    # print("answer：",answer)
    # print("cl_generate_text：",cl_generate_text)
    # -----------------文本题---------------------------------------------------------------------------------------------

    # -----------------选择题---------------------------------------------------------------------------------------------
    # context = "Hoping to break their current losing streak the Cowboys played on home ground for an Interconference duel with the Jaguars. In the first quarter the Cowboys took the lead as kicker David Buehler hit a 34-yard field goal. But they fell behind with QB David Garrard getting a 10-yard TD pass to WR Mike Sims-Walker. In the second quarter, the Cowboys struggled further with Garrard finding TE Marcedes Lewis on a 42-yard TD pass, then in the third quarter he found WR Mike Thomas on a 15-yard TD pass, and then he found Lewis again on a 9-yard TD pass. The Cowboys responded in the 4th quarter with RB Marion Barber getting a 1-yard TD run. But the Jaguars scored again with Garrard scrambling 2 yards to the endzone for a touchdown. The Cowboys replied with QB Jon Kitna making an 8-yard TD pass to TE Jason Witten."
    # context = "Find the order of the factor group (Z_11 x Z_15)/(<1, 1>).\n(A).1\n(B).2\n(C).5\n(D).11\n"
    # context = "Find the maximum possible order for an element of S_n for n = 10.\n(A).6(B).12(C).30(D).105"
    context = "Find the order of the factor group (Z_11 x Z_15)/(<1, 1>).\n(A).0\n(B).4\n(C).2\n(D).1\n"
    # context = "French senior civil servant arrested on suspicion of spying for North Korea\n\nNovember 27, 2018 by Joseph Fitsanakis\n\nA senior civil servant in the upper house of the French parliament has been arrested on suspicion of spying for North Korea, according to prosecutors. The news of the suspected spy’s arrest was first reported on Monday by Quotidien, a daily politics and culture show on the Monaco-based television channel TMC. The show cited “a judicial source in Paris” and said that France’s domestic security and counterintelligence agency, the General Directorate for Internal Security (DGSI), was in charge of the espionage case.\n\nThe senior administrator has been identified as Benoit Quennedey, a civil servant who liaises between the French Senate and the Department of Architecture and Heritage, which operates under France’s Ministry of Culture. Quennedey was reportedly detained on Sunday morning and his office in the French Senate was raided by DGSI officers on the same day. Quotidien said that he was arrested on suspicion of “collecting and delivering to a foreign power information likely to subvert core national interests”. The report did not provide specific information about the type of information that Quennedey is believed to have passed to North Korea. It did state, however, that a counterintelligence investigation into his activities began in March of this year.\n\nQuennedey is believed to be the president of the Franco-Korean Friendship Association, the French branch of a Spanish-based organization that lobbies in favor of international support for North Korea. Korea Friendship Association branches exist in over 30 countries and are believed to be officially sanctioned by Pyongyang. They operate as something akin to the pre-World War II Comintern (Communist International), a Moscow-sanctioned international pressure group that advocated in favor of Soviet-style communism around the world. French media reported on Monday that Quennedey traveled extensively to the Korean Peninsula in the past decade and has written a French-language book on North Korea. News reports said that the French President Emmanuel Macron had been made aware of Quennedey’s arrest. The senior civil servant faces up to 30 years in prison if found guilty of espionage.\n\n► Author: Joseph Fitsanakis | Date: 27 November 2018 | Permalink\n\n"
    # context = "If you had a bar of chocolate and then your friend gave you another bar of the same chocolate, how many bars of chocolate do you have now?\n(A).0(B).1(C).2(D).3"
    # context = "As at most other universities, Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one-page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut."
    # context = "As a result, by 1206 Temüjin had managed to unite or subdue the Merkits, Naimans, Mongols, Keraites, Tatars, Uyghurs, and other disparate smaller tribes under his rule. It was a monumental feat for the \"Mongols\" (as they became known collectively). At a Khuruldai, a council of Mongol chiefs, Temüjin was acknowledged as \"Khan\" of the consolidated tribes and took the new title \"Genghis Khan\". The title Khagan was not conferred on Genghis until after his death, when his son and successor, Ögedei, took the title for himself and extended it posthumously to his father (as he was also to be posthumously declared the founder of the Yuan dynasty). This unification of all confederations by Genghis Khan established peace between previously warring tribes and a single political and military force under Genghis Khan."
    # context = "It was only the orbit of the planet Mercury that Newton's Law of Gravitation seemed not to fully explain. Some astrophysicists predicted the existence of another planet (Vulcan) that would explain the discrepancies; however, despite some early indications, no such planet could be found. When Albert Einstein formulated his theory of general relativity (GR) he turned his attention to the problem of Mercury's orbit and found that his theory added a correction, which could account for the discrepancy. This was the first time that Newton's Theory of Gravity had been shown to be less correct than an alternative."
    # context = "In the past, teachers have been paid relatively low salaries. However, average teacher salaries have improved rapidly in recent years. US teachers are generally paid on graduated scales, with income depending on experience. Teachers with more experience and higher education earn more than those with a standard bachelor's degree and certificate. Salaries vary greatly depending on state, relative cost of living, and grade taught. Salaries also vary within states where wealthy suburban school districts generally have higher salary schedules than other districts. The median salary for all primary and secondary teachers was $46,000 in 2004, with the average entry salary for a teacher with a bachelor's degree being an estimated $32,000. Median salaries for preschool teachers, however, were less than half the national median for secondary teachers, clock in at an estimated $21,000 in 2004. For high school teachers, median salaries in 2007 ranged from $35,000 in South Dakota to $71,000 in New York, with a national median of $52,000. Some contracts may include long-term disability insurance, life insurance, emergency/personal leave and investment options. The American Federation of Teachers' teacher salary survey for the 2006-07 school year found that the average teacher salary was $51,009. In a salary survey report for K-12 teachers, elementary school teachers had the lowest median salary earning $39,259. High school teachers had the highest median salary earning $41,855. Many teachers take advantage of the opportunity to increase their income by supervising after-school programs and other extracurricular activities. In addition to monetary compensation, public school teachers may also enjoy greater benefits (like health insurance) compared to other occupations. Merit pay systems are on the rise for teachers, paying teachers extra money based on excellent classroom evaluations, high test scores and for high success at their overall school. Also, with the advent of the internet, many teachers are now selling their lesson plans to other teachers through the web in order to earn supplemental income, most notably on TeachersPayTeachers.com"
    question = "Which option is correct?"

    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer("### Context:\n")["input_ids"] + context_ids \

    padding_token = tokenizer('\n', add_special_tokens=False)["input_ids"]
    input_ids = pad_sequence(input_ids,510,padding_token)
    lm_targets = tokenizer("\n### Question:\n", add_special_tokens=False)["input_ids"] + question_ids \
                 + tokenizer("\n### Answer:\n", add_special_tokens=False)["input_ids"]
    inputs = {"input_ids":torch.tensor(input_ids).unsqueeze(0).to(model.device),
              "lm_targets":torch.tensor(lm_targets).unsqueeze(0).to(model.device)}
    answer = model.lm_inference(inputs,segment_size=510)
    answer = tokenizer.decode(answer, skip_special_tokens=True)
    print(inputs)
    cl_generate_text = model.cl_inference(inputs, 510)
    cl_generate_text = tokenizer.decode(cl_generate_text, skip_special_tokens=True)
    print(cl_generate_text)
    instruction_dataset_name = 'squad'
    if os.path.exists(f'../{instruction_dataset_name}_test_instruction_dataset.json'):
        with open(f'../{instruction_dataset_name}_test_instruction_dataset.json', 'r', encoding='utf-8') as f:
            examples_list = json.load(f)
    # print(examples_list)
    print(tokenizer.decode(input_ids, skip_special_tokens=True))
    print(tokenizer.decode(lm_targets, skip_special_tokens=True))
    print("-------------------------------------")
    print("answer：",answer)
    # print("cl_generate_text：",cl_generate_text)
    # # text = "Gone are the days when America’s standing in the world was contrasted primarily with that of the Soviet Union. Instead, the United States and China now compete to be the more favored world power.\n\nThe U.S. and China engender roughly the same level of goodwill. China is particularly well-liked in Latin America and the Middle East, while the U.S. fares better in Europe and the Asia-Pacific region.\n\nHowever, America’s weakening image in many nations has taken a toll on the country’s once-solid lead over China. And China’s own favorability has strengthened in recent years in Canada, Australia, Lebanon and Turkey.\n\nSince the most recent year Pew Research Center polled in 36 nations – 2014, 2015 or 2016, depending on the country – the number of nations in which the U.S. holds a competitive advantage in favorability over China has halved, from 25 to 12. (Differences of less than 6 percentage points are considered ties.) Whereas the U.S. once had a 12-point lead over China in terms of a global median, that lead has shrunk in 2017 to 2 points.\n\nIn six nations – Spain, Mexico, Turkey, Australia, Peru and Senegal – the dynamic between the two superpowers has flipped, with China overtaking the U.S. in favorability.\n\nAnd the United States’ once-significant lead over China in popularity has fallen to a virtual tie in another seven countries: Kenya, Germany, France, Brazil, Sweden, the UK and Canada.\n\nMeanwhile, in 12 nations, people view America more favorably than they do China: Vietnam, Israel, the Philippines, South Korea, Poland, Hungary, Italy, Ghana, Japan, South Africa, Colombia and India.\n\nA quarter-century after the collapse of the Soviet Union, Russia is viewed far less favorably than either the U.S. or China in most of the world, though America’s recent steep decline in image has improved Russia’s standing compared with that of the U.S.\n\nAmerica’s edge over Russia has contracted by more than 20 percentage points in 15 out of the 33 nations for which Pew Research Center has trend data on favorability toward Russia. These include Spain, France, Chile, Brazil, Italy, Australia and Tanzania.\n\nThe narrowing of the U.S.-Russia favorability gap is most striking in Mexico, where the 42-point advantage held by the U.S. over Russia in 2015 is all but gone. Mexicans now view the U.S. and Russia roughly the same.\n\n"
    # # text = "All were either rejected outright by Metro — based on its advertising guidelines that prohibit ads that are “issues-oriented” or “intended to influence members of the public regarding an issue on which there are varying opinions” — or were retroactively pulled from stations, trains and buses after riders complained.\n\nAD\n\nAD\n\n“This case highlights the consequences of the government’s attempt to suppress all controversial speech on public transit property,” Arthur Spitzer, legal director of the ACLU-DC and lead counsel in the case, said in a statement. “The First Amendment protects the speech of everyone from discriminatory government censorship, whether you agree with the message or not.”\n\nThe ACLU said that Metro is enforcing its advertising guidelines capriciously, and that the prohibitions outlined in the guidelines are far too broad and wide-reaching.\n\nThe organization noted that any advertisement could potentially violate Metro’s policy, and that the transit agency has allowed other advertisements for organizations or issues that could be polarizing.\n\nAD\n\n“By rejecting these ads and accepting ads from gambling casinos, military contractors, and internet sex apps, the [Washington Metropolitan Area Transit Authority] showed just how subjective its ban is,” the statement said.\n\nAD\n\n“WMATA’s policy is an attempt to silence anyone who tries to make you think. Any one of these advertisements, had it passed the WMATA’s censor, would have been the subject of someone’s outraged call to the WMATA,” the ACLU added. “The First Amendment doesn’t, and shouldn’t, tolerate that kind of impoverishment of our public conversation. Not even in the subway.”\n\nCarafem, the company whose advertisements for abortion pills were rejected by Metro, said it is a health-care provider, not an advocacy group, meaning its ads do not violate Metro’s policy.\n\nAD\n\n“The abortion pill is, of course, both FDA-approved and accepted by the American Medical Association. We are a healthcare provider, not an advocacy group. Metro’s ban of our ads claimed that they were ‘issue-oriented’ and ‘provided a medical statement which can only be accepted from a government health association,’ ” Melissa Grant, Carafem’s chief operations officer, said in a statement. “This is obviously inaccurate — we’re publicizing our services like any other health care provider."
    # text = " a With Red Hat, IBM to become the leading hybrid cloud provider Watch Now\n\nAfter IBM acquired Red Hat, I suggested IBM paid $34 billion for the Linux power so it could become a hybrid-cloud power. With the news that Red Hat will acquire NooBaa, a hybrid-cloud, data-storage company, it's become clearer than ever that the IBM-Red Hat deal is all about the hybrid cloud.\n\nNooBaa is a software-defined storage company. Its primary program of the same name is open-source software, which puts a virtual layer over private and public clouds storage resources.\n\nAlso: IBM: Chip making is hitting its limits, but our techniques could solve that\n\nIt's made up of three components: First, there's an access node, which handles the data chunking, deduplication, compression and encryption between storage resources; next, there's a storage daemon, which presents server storage as storage nodes; and finally, there's a virtual machine (VM) based core for data placement, self-healing, and monitoring. The Access nodes and storage daemons make up a data plane, while the core provides its control plane.\n\nAlso: How IBM Watson is revolutionizing 10 industries TechRepublic\n\nSo, what does all mean for customers? It's multi-cloud storage management, which enables allows you to manage, deploy, and migrate data storage across private and major public clouds. This includes Alibaba, AWS, Azure, and Google Cloud.\n\nIt's easy to see why Red Hat values this. It gives their customers a way to manage storage without sweating the details across multiple platforms.\n\nAs Ranga Rangachari, Red Hat's vice president of Storage and Hyperconverged Infrastructure, said in a statement:\n\n\"Data portability is a key imperative for organizations building and deploying cloud-native applications across private and multiple clouds. NooBaa's technologies will augment our portfolio and strengthen our ability to meet the needs of developers in today's hybrid and multicloud world. We are thrilled to welcome a technical team of nine to the Red Hat family as we work together to further solidify Red Hat as a leading provider of open hybrid-cloud technologies.\"\n\nRelated stories:\n\ "
    # cl_gen_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    # print(text)





