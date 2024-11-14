
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
from torch import cosine_similarity

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

# 选择一个模型的tokenizer（如BERT模型）
work_dir = "../compressLLM_multi_lora_510_ratio_lm&cl"
with open(work_dir + f'/config.json') as f:
    config =json.load(f)

config["data_config"]["model_id"] = "../../../models/TinyLlama/TinyLlama_v1.1"
world_size = torch.cuda.device_count()
tokenizer = AutoTokenizer.from_pretrained(config["data_config"]["model_id"], token=config["data_config"]["hf_token"])
model = AutoModelForCausalLM.from_pretrained(config["data_config"]["model_id"], token=config["data_config"]["hf_token"])


correct_tokens = 0
# 输入文本
# 41
# text1 = "Hindus, who urged disciplining of a Northern Ireland Catholic priest who reportedly linked yoga to Satan, are disheartened over the Church’s silence on the issue.\n\nFather Reverend Roland Colhoun, a priest at Waterside Parish of Roman Catholic Diocese of Derry in Northern Ireland, as reported by Derry Journal, “warned parishioners against taking part in yoga” while saying mass in Drumsurn recently. “Yoga is certainly a risk. There’s the spiritual health risk”, Journal quoted him.\n\nUK’s The Independent quoted him as saying “Yoga leads to Satan” and that he fears it could lead to “The Kingdom of Darkness”. “It’s a slippery slope from yoga to Satan”, RT channel said quoting him.\n\nHindu statesman Rajan Zed, in a statement in Nevada (USA) today, said that reports suggested that neither the Vatican and nor the Diocese of Derry Bishop, Most Reverend Donal McKeown, had offered apology on the actions of the priest who just trashed a highly revered and respected ancient practice.\n\nZed, who is President of Universal Society of Hinduism, stressed that although introduced and nourished by Hinduism, yoga was a world heritage and liberation powerhouse being utilized by millions of people of various faith traditions worldwide for achieving physical and mental fitness.\n\nWhy this Derry priest was unnecessarily disparaging yoga whom Patanjali described as “a methodical effort to attain perfection” and about whom US National Institutes of Health indicated: yoga may help one to feel more relaxed, be more flexible, improve posture, breathe deeply, and get rid of stress? Rajan Zed asked.\nZed further said that Bishop McKeown, whose Diocese of Derry was directly responsible for the actions of this priest, should apologize to the hurt worldwide yoga community and clearly state where he stood on this issue. Moreover, Vatican should also clarify its stand on yoga, honestly and transparently.\nYoga, referred as “a living fossil”, was a mental and physical discipline, for everybody to share and benefit from, whose traces went back to around 2,000 BCE to Indus Valley civilization, Rajan Zed pointed out and added that yoga was the repository of something basic in the human soul and psyche.",
# text2 = "Hindus, who urged disining of a Ireland Ireland religious Catholic who reportedly linked ygs to thean, are dis heartened over the Church’s silence on the issue.\n\nFather ReverendR Colhoun, a W at Waterside Parish of Roman Catholicedese of ofry in Northern Ireland, as reported by Denry Journal, “Iedionioners against taking part in y in” while saying mass in Drsurn recently. “Y has is certainly a risk. There’s the religious health risk”, Journal quoted him.\n\nStep’s The/ quoted him as saying “Y leads leads to satan” and that he fears it could lead to “The Kingdom of ofness”. “It’s a slipperyys from y\" to (an”, RT channel said quoting him.\n\nH Muslim statesman Rajan Zed, in a statement in Nevada ( USA) today, said that reports suggested that neither the Vatican and or theineese of derry,, Most Reverend Donal Mc Mown, had offered apology on the actions of the Francis who just tr\n a highly ored and respected practice practice.\n\nZed, who is President of the Society of H toism, stressed that although introduced and nourished by H byism, y\" was a world herity and releasedation powerhouse being utilized by millions of people of various faith traditions worldwide for’ieving physical and mental fitness.\n\nWhy this derry it wasitarilyagesaging y — whom Patanjali described as “a methodical effort to attain perfection” and about whom US Nationalom US of Health indicated: y  may help one to feel more relaxed, be more our, improve posture, breathe she, and get rid of stress? Rajan Zed asked.\n\nZed further said that Church Mckeown, whoseThese of derry was directly responsible for the actions of this apolog, should apologize to the hurt worldwide y community community and clearly state where he stood on this issue. issue, Vatican should also' its stand on y In, honestly and transparently.\n\nYoga, referring as “a living fossil”, was a mental and physical and, for everybody to share and benefit from, whose whose went back to around 2 ### Context:\n for everybody to share and benefit from, whose traces went back to"
text1 = "2017 has been a--packed year for Rare and and of Thieves, and as we position our sils towards that long- longed launch in early 2018, we’re also coming up on a full year of the Sea of Thists Teical Alp. It’s served the game incredibly well and given thousands upon thousands ofdding buccaneers a chance to hit the seas on Xbox One and Windows 10 PC – and we’ll soon be waving good to to the technical Al, in its current form, so this is your final chance to get involved!\n\nWhy sign up now if you haven’t already? Everyone who joins the Insider Programme by the cut-off date of Dec. 1 will be invited to the next play session – our biggest and best yet – with the latest brace of brand new features to test. go, as promised, all eligible insers are now getting their chance to jump in and play Sea of Th is before its release next year.\n\nIt was back in December 2016 that we announced the very first ical Al sessions session, following the kick-off of our Insider programme in November. In the space of a year we’ve scaled up from that first wave of 1,000 players to almost 200,000. We’ve debuted new features fromalletal sentinels to alone ships, welcomed your feedback and used it to balance and finetune all manner of things, honoring our promise to shape Sea of Thieves alongside its community. Now we’re preparing to welcome all those insiders still waiting to take to the waves, plus all new Insiders who sign up between now and Dec. 1.\n\nSo if you haven’t yet pounced on the possibility of playing playing of Thieves early, waste no more time and join the insider staff at the.seaofthieves.com/insider. Our Insiders receive regular updates from the development team through official, posts and newsletters, and of course get the chance to play the game and share their with with us as it continues to evolutionve.\n\nJoining the ins1 Programme is free – you just need to have an Xbox Live account (or set up"
text2 = "2017 has been a jam-packed year for Rare and Sea of Thieves, and as we angle our sails towards that long-awaited launch in early 2018, we’re also coming up on a full year of the Sea of Thieves Technical Alpha. It’s served the game incredibly well and given thousands upon thousands of budding buccaneers a chance to hit the seas on Xbox One and Windows 10 PC – and we’ll soon be waving goodbye to the Technical Alpha in its current form, so this is your final chance to get involved!\n\nWhy sign up now if you haven’t already? Everyone who joins the Insider Programme by the cut-off date of Dec. 1 will be invited to the next play session – our biggest and best yet – with the latest brace of brand new features to test. Yes, as promised, all eligible Insiders are now getting their chance to jump in and play Sea of Thieves before its release next year.\n\nIt was back in December 2016 that we announced the very first Technical Alpha sessions, following the kick-off of our Insider Programme in November. In the space of a year we’ve scaled up from that first wave of 1,000 players to almost 200,000. We’ve debuted new features from skeletal sentinels to solo ships, welcomed your feedback and used it to balance and finetune all manner of things, honoring our promise to shape Sea of Thieves alongside its community. Now we’re preparing to welcome all those Insiders still waiting to take to the waves, plus all new Insiders who sign up between now and Dec. 1.\n\nSo if you haven’t yet pounced on the possibility of playing Sea of Thieves early, waste no more time and join the Insider crew at www.seaofthieves.com/insider. Our Insiders receive regular updates from the development team through official forum posts and newsletters, and of course get the chance to play the game and share their thoughts with us as it continues to evolve.\n\nJoining the Insider Programme is free – you just need to have an Xbox Live account (or set up a new one) and be 18 years or older to sign the NDA. As Sea of Thieves is an online multiplayer game, Xbox Live Gold membership is also required when sailing the seas on Xbox One. Please remember that these Technical Alpha play sessions are purely for the benefit of our Insiders and are currently under NDA, so they can’t be streamed or captured!\n\nFor more information about the Insider Programme including some details on eligibility, see our FAQ. And stay tuned to official Sea of Thieves and Xbox social channels for an announcement of the big play session date coming up in December.\n\nSee you all on the seas!\n\n"
# 对文本进行分词并计算token的数量
tokens1 = torch.tensor(tokenizer(text1)["input_ids"]).squeeze(0)[:510]
tokens2 = torch.tensor(tokenizer(text2)["input_ids"]).squeeze(0)[:510]
with torch.no_grad():
    embeddings1 = model.model.embed_tokens(tokens1)  # 计算文本1的平均嵌入
    embeddings2 = model.model.embed_tokens(tokens2) # 计算文本2的平均嵌入
    correct_tokens += sum(1 for o,d in zip(tokens1, tokens2) if o == d)
# acc = cosine_similarity(embeddings1,embeddings2).mean()
print("embeddings1:", embeddings1)
print("embeddings2:", embeddings2)
# 计算两个文本平均嵌入的余弦相似度
# cos_sim = cosine_similarity(embeddings1,embeddings2).mean()
# print(cos_sim)

# print("文本相似度:", cos_sim)
print("acc2:", correct_tokens / len(tokens2))
# 11
# text1 = "An investment made by Kobe Bryant has yielded more than 30 times its money in fewer than four and a half years.\n\nOn Tuesday, Coca-Cola announced it had purchased a minority stake in sports drink BodyArmor.\n\nBryant made his first investment in the brand, for roughly 10 percent of the company, in March 2014, putting in roughly $6 million over time. Based on the valuation of the Coca-Cola deal, his stake is now worth approximately $200 million, sources told ESPN.\n\nBryant is now the fourth-largest investor in the brand, marketed as a healthier competitor to Gatorade, behind the brand's co-founder Mike Repole, Coca-Cola and Keurig Dr Pepper. When Bryant invested in BodyArmor, the brand had just come off a year of $10 million in sales. BodyArmor is projected to top $400 million in sales in 2018.\n\nKobe Bryant's other March 2014 investment yielded an Oscar for \"Dear Basketball.\" Kevin Winter/Getty Images\n\nBryant, who earned $328 million on the court in his 20-year NBA career and a similar amount off the court over that time, announced his investment in BodyArmor on the same day he announced the start of his new company, Kobe Inc. He since has formed a $100 million joint venture investment firm with entrepreneur Jeff Stibel and started his own production company, Granity Studios, which won an Oscar in 2018 for best animated short for his \"Dear Basketball\" film.\n\nAs part of their endorsement deals, many athletes had equity stakes in BodyArmor. Sources told ESPN that as many as a dozen superstar athletes could also have stakes in BodyArmor worth more than $1 million, including James Harden, Dustin Johnson and Andrew Luck.\n\nCoca-Cola's acquisition is the biggest story in the business of sports drinks since December 2000, when PepsiCo acquired Quaker Oats, which included Gatorade. The deal puts BodyArmor in Coke's powerful distribution network, on their delivery trucks throughout most of the United States.\n\nThis is the second time Repole has sold a company to Coca-Cola. In 2007, Glaceau, a company he co-founded with the smartwater and vitaminwater brands, sold to Coke for $4.1 billion.\n\nBryant's return is the biggest return for a modern-day athlete in the business world in some time. LeBron James made $30 million from a small stake in Beats by Dre when it sold to Apple in May 2014. James and his business partner Maverick Carter put less than $1 million into fast-casual pizza chain Blaze in 2012. That investment is now worth approximately $40 million.\n\n"
# text2 = "An investment made by K Bry Bryant has hased more than 30 times its money in fewer than four and a half years.\n\nOn Tuesday, Coca-Cola announced it had purchased a minority stake in sports drink.Ar..\n\nBryant made his first investment in the brand, for roughly 10 percent of the company, in March 2014, putting in roughly $6 million over time. based on the valueation of the Coca-Cola deal, his stake is now worth approximately $200 million, sources told ESPN.\n\nBryant is now the fourth- largestest investor in the brand, marketed as a healthier competitor to Gatorade, behind the brand's co-founder Mike Re Re, Coca-Cola and Keurig Dr Pepper. Whenryant invested in\nAr brand, the brand had just come off a year of $10 million in sales.\nAr is is projected to top $400 million in sales in 2018.\n\nKBanant's other March 2014 investment.ed an Oscar for \"Dear basketball.\" Kevin Kevin/Getty Images\n\nBryant, who earned $328 million on the court in his 20-year NBA career and a similar amount off the court over that time, announced his investment in bodyAror on the same day he announced the start of his new company, K. Inc. He since has formed a $100 million joint venture investment firm with entrepreneur Jeff Stibel and started his own production company, Grandity Studios, which won an Academy in 2018 for best animated short for his \"Dear basketball\" film.\n\nAs part of their endorsement deals, many athletes had equity stakes in bodyArAr. Sources told ESPN that as many as a dozen super stars athletes could also have stakes in bodyArAr worth more than $1 million, including James Hard, Dustin Johnson and Andrew Luck.\n\nCoca-ocaa's acquisition is the biggest story in the business of sports drinks since December 2000, when Pe hisCo acquired Quaker Oats, which included Gatorade. The"
# 95
# text1 = "California State Assemblyman Reggie Jones-Sawyer (D-Paramount) kicked off Tuesday’s confirmation hearing for newly-appointed state Attorney General Xavier Becerra by predicting a “legal war” between California and the Trump administration.\n\nThe San Diego Union-Tribune and other outlets reported that Jones-Sawyer said California faced a “long, legal war” with President-elect Donald Trump after he ran “the most xenophobic campaign in modern history.”\n\n“The incoming president ran the most xenophobic campaign in modern history,” says Assemblyman @JonesSawyer59 in opening of Becerra A hearing — John Myers (@johnmyers) January 10, 2017\n\nBecerra, who is currently a U.S. congressman and prominent member of the Democratic caucus, was nominated to take over as California Attorney General by Gov. Jerry Brown after Kamala Harris won a U.S. Senate seat. She was sworn in last week. Thus far, he has built his candidacy for the job around a promise to resist the policies of the Trump administration, especially on immigration. Separately, the state legislature has retained former U.S. Attorney General Eric Holder for the same purpose — though the constitutionality of that hire is being challenged by Republican State Assemblyman Kevin Kiley (R-Roseville).\n\nBrown introduced Becerra at the State Assembly’s hearing by warning that “there are big battles ahead.” In his own remarks, Becerra voiced an apparently newfound interest in states’ rights, vowing to resist “federal intrusion” in California: “You will find me being as aggressive as possible working with all of you to figure out ways that we can make sure there is no federal intrusion in areas that are really left to the state in the U.S. Constitution,” he said, according to the Los Angeles Times.\n\nImmigration, as Democrats have repeatedly noted, is an area under the exclusive jurisdiction of the federal government.\n\nThe State Assembly panel voted 6-3 on party lines to recommend Becerra’s appointment, the Times reported.\n\nJoel B. Pollak is Senior Editor-at-Large at Breitbart News. He was named one of the “most influential” people in news media in 2016. His new book, How Trump Won: The Inside Story of a Revolution, is available from Regnery. Follow him on Twitter at @joelpollak.\n\n",
# text2 = "California State Assemblyman Reg people Jones-Sawyer (D-),ount) kicked off Tuesday’s confirmation hearing for new-appointed state Attorney General X X Becerra by predicting a “ legal war” between California and the Trump administration.\n\nThe San Diego Union-Tribune and other outlets reported that Jones-Saw has said California faced a “long, legal war” with President-elect Donald Trump after he ran “the most xenophobic campaign in modern history.”\n\n“TheThe president ran the most xenophobic campaign in modern history,” says Assemblyman @JonesSawyer59 in opening of Becerra A hearing — John Myers (@j onemyers) January 10, 2017\n\nBecerra, who is currently a U.S. congressman and prominent member of the Democratic caucus, was nominated to take over as California Attorney General by Gov. Jerry Brown after Kamala Harris won a U.S. Senate seat. She was sworn in last week. thus far, he has built his hisacy for the job around a promise to to the policies of the Trump administration, especially on immigration. Separately, the state legislature has retain former U.S. Attorney Generalic Hale for the same purpose — though the constitution that of that hire is being challenged by Republican State Assemblyman Kevin Kiley (R-Roseville).\n\nBrown introduced Becerra at the State Assembly’s hearing by warning that “there are big battles ahead.” In his own remarks, Becerra areiced an apparently newfound interest in states’ rights, vowing to resist “fFal intrusion” in California: “You will find me being as aggressive as possible working with all of you to figure out ways that we can make sure there is no federal intrusion in areas that are really left to the state in the U.S. Constitution,” he said, according to the Los Angeles Times.\n\nImmigration, as Democrats have noted noted, is an area under the exclusive exclusive juris of the federal government.\n\nThe State Assembly panel voted 6-3 on party lines to recommend Becerra’s appointment, the Times reported.\n\n Joel B. Polur is senior editor"
