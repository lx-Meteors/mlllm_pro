import logging
import pdb
import queue
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import math
import transformers

sys.path.append('/mnt/zhaorunsong/lx/compress/meteor/')
# noinspection PyUnresolvedReferences
from modify_code import modify_llama


class TripleLinearLoraLayer(nn.Module):
    def __init__(self, in_features, out_features, r_cl=16, r_lm=16, r_cl_prime=16, scale=1.0, weight=None):
        super(TripleLinearLoraLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = 2  # Scaling factor

        # 原始权重矩阵 W
        self.weight = nn.Parameter(weight, requires_grad=False)

        # 压缩 token 的 LoRA 模块参数 (A_cl, B_cl)
        self.lora_A_cl = nn.Parameter(torch.zeros((in_features, r_cl), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_cl = nn.Parameter(torch.zeros((r_cl, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

        # LLM token 的 LoRA 模块参数 (A_lm, B_lm)
        self.lora_A_lm = nn.Parameter(torch.zeros((in_features, r_lm), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_lm = nn.Parameter(torch.zeros((r_lm, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

        # 额外的压缩 token 的 LoRA 模块参数 (A_cl', B_cl')
        self.lora_A_cl_prime = nn.Parameter(torch.zeros((in_features, r_cl_prime), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_cl_prime = nn.Parameter(torch.zeros((r_cl_prime, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

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
    def __init__(self, in_features, out_features, padding_idx, r_cl=128, r_lm=128, r_cl_prime=128, scale=1.0, weight=None):
        super(TripleEmbeddingLoraLayer, self).__init__()
        self.num_embeddings = in_features
        self.embedding_dim = out_features
        self.padding_idx = padding_idx
        self.scale = 2  # Scaling factor

        # 原始权重矩阵 W
        self.weight = nn.Parameter(weight, requires_grad=False)

        # 压缩 token 的 LoRA 模块参数 (A_cl, B_cl)
        self.lora_A_cl = nn.Parameter(torch.zeros((in_features, r_cl), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_cl = nn.Parameter(torch.zeros((r_cl, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

        # LLM token 的 LoRA 模块参数 (A_lm, B_lm)
        self.lora_A_lm = nn.Parameter(torch.zeros((in_features, r_lm), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_lm = nn.Parameter(torch.zeros((r_lm, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

        # 额外的压缩 token 的 LoRA 模块参数 (A_cl', B_cl')
        self.lora_A_cl_prime = nn.Parameter(torch.zeros((in_features, r_cl_prime), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B_cl_prime = nn.Parameter(torch.zeros((r_cl_prime, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)

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
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, enabled_lora):
        if enabled_lora:
            result = F.linear(x, self.weight)
            result += self.scale * (x @ self.lora_A @ self.lora_B)
        else:
            result = F.linear(x, self.weight)
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
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B)

    def forward(self, x, enabled_lora):
        if enabled_lora:
            result = F.embedding(x, self.weight, self.padding_idx)
            after_A = F.embedding(x, self.lora_A, self.padding_idx)
            result += self.scale * (after_A @ self.lora_B)
        else:
            result = F.embedding(x, self.weight, self.padding_idx)
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
        self.mem_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((mem_size, config.hidden_size)), requires_grad=True)
        self.special_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((2, config.hidden_size)), requires_grad=True)
        self.head_num = head_num

        self.compress_head = nn.Linear(config.hidden_size, head_num*config.vocab_size, bias=False, device=f"cuda:{device_rank}",
                                            dtype=self.model.model.embed_tokens.weight.dtype)

        # self.compress_head = nn.Sequential(
        #     nn.Linear(config.hidden_size, head_num*128, bias=False, device=f"cuda:{device_rank}", dtype=self.model.model.embed_tokens.weight.dtype),
        #     nn.Linear(head_num*128, head_num*config.vocab_size, bias=False, device=f"cuda:{device_rank}", dtype=self.model.model.embed_tokens.weight.dtype)
        #     )
        mean = torch.mean(self.model.model.embed_tokens.weight).item()
        std = torch.std(self.model.model.embed_tokens.weight).item()
        nn.init.normal_(self.mem_tokens, mean=mean, std=std)
        nn.init.normal_(self.special_tokens, mean=mean, std=std)

    def forward(self,inputs):
        # ->LlamaForCausalLM->LlamaModel->embed_tokens
        if self.task_config["use_multi_lora"]:
            mask = {"lm_mask": torch.ones_like(inputs['input_ids'])}
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"], enabled_lora=False)
        else:
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"], enabled_lora=False)
        bsz, seq_len, emb_size = inputs_embeds.size()
        mem_size = self.mem_tokens.size(0)
        expand_mem = self.mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
        encode_inputs_embeds = torch.cat([inputs_embeds,expand_mem],dim=1)

        # [1,seq_len]
        position_ids = torch.arange(1,seq_len+1,device=inputs_embeds.device).unsqueeze(0)
        # [1,mem_size]
        mem_position_ids = torch.arange((self.head_num+1)//2, self.head_num*mem_size+1, step=self.head_num, device=inputs_embeds.device).unsqueeze(0)
        # [1,seq_len+mem_size]
        encode_position_ids = torch.cat([position_ids,mem_position_ids],dim=1)

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
                    mask = mask,
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
                    enabled_lora=False
                )

        hidden_states = outputs.hidden_states[-1]
        
        # [B,mem_size,emb_size]
        mem_hidden = hidden_states[:,-mem_size:]
        # [B,seq_len,vocab_size]
        original_logits = outputs.logits[:,:seq_len]

        tot_loss = 0
        tot_task = 0
        loss_info = {}


        # compress loss：压缩的是输入的context，并不是prompt和answer
        if self.task_config["use_compress_loss"]:
            # print("compress_targets will be used")
            # [B,mem_size,emb_size] -> [B,mem_size,head_num*vocab_size]
            logits =  self.compress_head(mem_hidden)

            # extract original logits
            # [B,mem_size,head_num*vocab_size] -> [B,tot_Seg_len,V] -> [B,seq_len,V]
            logits = logits.reshape(bsz,mem_size*self.head_num,self.vocab_size)
            logits = logits[:,:seq_len,:]

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
                if original_logits.shape[1] != inputs["instruction_target"].shape[1]: # if only <eos> in next segment, they will be equal.
                    # no token after <eos> [context + prompt + answer 510]
                    original_logits = original_logits[:,:-1]
                logits = original_logits.contiguous().view(-1, self.vocab_size)
                inputs["instruction_target"] = inputs["instruction_target"].contiguous().view(-1).to(logits.device)

                lm_loss = self.loss_fct(logits, inputs["instruction_target"])
                loss_info["lm_loss"] = lm_loss.item()
                return {"loss":lm_loss, "loss_info":loss_info}

            if self.task_config["use_multi_lora"]:
                mask = {"lm_mask": torch.ones_like(inputs['lm_targets'][:,:-1])}
                # [B,seq_len-1] -> [B,seq_len-1,E]
                lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'][:,:-1], enabled_lora=False)
            else:
                lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'][:, :-1], enabled_lora=False)
            
            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)

            # todo: 1.将mem_hidden设置为0, .detach()
            #  [B,mem_size,E];     [B,1,E];      [B,seq_len-1,E]
            lm_emb = torch.cat([mem_hidden, expand_lm_token,lm_target_emb],dim=1)

            latter_position_ids = torch.arange(seq_len,seq_len+1+lm_target_emb.size(1),device=inputs_embeds.device).unsqueeze(0)
            lm_position_ids = torch.cat([mem_position_ids,latter_position_ids],dim=1)

            # make three masks
            if self.task_config["use_multi_lora"]:
                mask = make_masks(torch.cat([expand_lm_token,lm_target_emb],dim=1), mem_hidden, compress_prime_token=True)
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
                        enabled_lora=True
                    )

            # [B,mem_size+S,V] -> [B,S,V]
            logits = outputs.logits[:,mem_size:]

            # here, we cat the whole seq's logits
            logits = torch.cat([original_logits, logits[:,1:]], dim=1)
            logits = logits.contiguous().view(-1, self.vocab_size)
            inputs["instruction_target"] = inputs["instruction_target"].contiguous().view(-1).to(logits.device)

            lm_loss = self.loss_fct(logits, inputs["instruction_target"])
            loss_info["lm_loss"] = lm_loss.item()
            tot_loss += lm_loss
            tot_task += 1

        loss = tot_loss/tot_task

        return {"loss":loss, "loss_info":loss_info}

    def lm_inference(self,inputs,segment_size):
        # ->LlamaForCausalLM->LlamaModel->embed_tokens
        if self.task_config["use_multi_lora"]:
            mask = {"lm_mask": torch.ones_like(inputs["input_ids"])}
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"], mask)
        else:
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"], enabled_lora=False)
        bsz, seq_len, emb_size = inputs_embeds.size()
        mem_size = self.mem_tokens.size(0)

        # [1,seq_len]
        position_ids = torch.arange(1,seq_len+1,device=inputs_embeds.device).unsqueeze(0)


        if inputs['lm_targets'] is None:
            generate_text = []
            past_key_values = None
            next_inputs_embeds = inputs_embeds.clone()
            next_position_ids = position_ids.clone()
            
            for i in range(4096):
                if self.task_config["use_multi_lora"]:
                    mask = {"lm_mask": torch.ones_like(next_inputs_embeds)}
                    if "wo_pe" in self.task_config:
                        out = self.model(inputs_embeds=next_inputs_embeds, past_key_values=past_key_values, use_cache=True, mask=mask)
                    else:
                        out = self.model(position_ids=next_position_ids, inputs_embeds=next_inputs_embeds, past_key_values=past_key_values, use_cache=True, mask=mask)
                else:
                    if "wo_pe" in self.task_config:
                        out = self.model(inputs_embeds=next_inputs_embeds, past_key_values=past_key_values,
                                         use_cache=True)
                    else:
                        out = self.model(position_ids=next_position_ids, inputs_embeds=next_inputs_embeds,
                                         past_key_values=past_key_values, use_cache=True, enabled_lora=False)
                # [B,S,V] -> [B,V]
                logit = out.logits[:, -1]
                past_key_values = out.past_key_values
                # [B,V]->[B]
                next_token_id = torch.argmax(logit, dim=-1)

                # [B]->[B,E]->[B,1,E]
                if self.task_config["use_multi_lora"]:
                    mask = {"lm_mask": torch.ones_like(next_token_id)}
                    next_inputs_embeds = self.model.model.embed_tokens(next_token_id, mask).unsqueeze(1).to(inputs_embeds.device)
                else:
                    next_inputs_embeds = self.model.model.embed_tokens(next_token_id, enabled_lora=False).unsqueeze(1).to(inputs_embeds.device)
                next_position_ids = next_position_ids[:,-1:]+1 # [1, seq_len]/[1,1] -> [1,1]
                generate_text.append(next_token_id.item())
                if next_token_id.item() == 2: # eos
                    return generate_text
                if next_position_ids.item()>segment_size:
                    expand_mem = self.mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
                    encode_inputs_embeds = expand_mem

                    # [1,mem_size]
                    mem_position_ids = torch.arange((self.head_num+1)//2, segment_size+1, step=self.head_num, device=inputs_embeds.device).unsqueeze(0)
                    # [1,seq_len+mem_size]
                    encode_position_ids = torch.cat([position_ids,mem_position_ids],dim=1)

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
                                enabled_lora=False
                            )

                    hidden_states = outputs.hidden_states[-1]
                    
                    # [B,mem_size,emb_size]
                    mem_hidden = hidden_states[:,-mem_size:]
                    
                    # [1,E] -> [1,1,E] -> [B,1,E]
                    expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)
                    
                    #                  [B,mem_size,E];     [B,1,E]; 
                    lm_emb = torch.cat([mem_hidden, expand_lm_token],dim=1)

                    #                              [1,mem_size];    [1,1];
                    lm_position_ids = torch.cat([mem_position_ids,next_position_ids-1],dim=1)

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
                                             past_key_values=past_key_values, use_cache=True, enabled_lora=True)
                    past_key_values = out.past_key_values

                    # next_token_id and next_position_ids don't be changed here.

        else:
            if self.task_config["use_multi_lora"]:
                mask = {"lm_mask": torch.ones_like(inputs['lm_targets'])}
                after_embeds = self.model.model.embed_tokens(inputs['lm_targets'], mask)
            else:
                after_embeds = self.model.model.embed_tokens(inputs['lm_targets'], enabled_lora=False)
            expand_mem = self.mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
            encode_inputs_embeds = torch.cat([inputs_embeds,expand_mem],dim=1)

            # [1,mem_size]
            mem_position_ids = torch.arange((self.head_num+1)//2, segment_size+1, step=self.head_num, device=inputs_embeds.device).unsqueeze(0)
            # [1,seq_len+mem_size]
            encode_position_ids = torch.cat([position_ids,mem_position_ids],dim=1)

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
                        enabled_lora=False
                    )

            hidden_states = outputs.hidden_states[-1]

            # [B,mem_size,emb_size]
            mem_hidden = hidden_states[:,-mem_size:]

            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)

            #                     [B,mem_size,E];     [B,1,E];      [B,seq_len-1,E]
            lm_emb = torch.cat([mem_hidden, expand_lm_token,after_embeds],dim=1)

            after_len = expand_lm_token.size(1) + after_embeds.size(1)
            after_position_ids = torch.arange(segment_size, segment_size+after_len, device=inputs_embeds.device).unsqueeze(0)
            #                              [1,mem_size];    [1,seq_len];
            lm_position_ids = torch.cat([mem_position_ids,after_position_ids],dim=1)


            generate_text = []
            past_key_values = None
            next_inputs_embeds = lm_emb.clone()
            next_position_ids = lm_position_ids.clone()
            if self.task_config["use_multi_lora"]:
                mask = make_masks(torch.cat([expand_lm_token,after_embeds],dim=1), mem_hidden, compress_prime_token=True)
            for i in range(4096):
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
                                         use_cache=True,
                                         enabled_lora=True)
                # [B,S,V] -> [B,V]
                logit = out.logits[:, -1]
                past_key_values = out.past_key_values
                # [B,V]->[B]
                next_token_id = torch.argmax(logit, dim=-1)

                # [B]->[B,E]->[B,1,E]
                if self.task_config["use_multi_lora"]:
                    mask = {"lm_mask": torch.ones_like(next_token_id)}
                    next_inputs_embeds = self.model.model.embed_tokens(next_token_id, mask).unsqueeze(1).to(inputs_embeds.device)
                    mask = {"lm_mask": torch.ones_like(next_inputs_embeds)}
                else:
                    next_inputs_embeds = self.model.model.embed_tokens(next_token_id, enabled_lora=True).unsqueeze(1).to(inputs_embeds.device)
                # todo: 不是很理解这里每次都是[1,1]和+1的作用
                next_position_ids = next_position_ids[:,-1:]+1 # [1, seq_len]/[1,1] -> [1,1]
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
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"], enabled_lora=False)
        bsz, seq_len, emb_size = inputs_embeds.size()
        mem_size = self.mem_tokens.size(0)
        expand_mem = self.mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
        encode_inputs_embeds = torch.cat([inputs_embeds, expand_mem],dim=1)

        # [1,seq_len]
        position_ids = torch.arange(1, seq_len+1, device=inputs_embeds.device).unsqueeze(0)
        # [1,mem_size]
        mem_position_ids = torch.arange((self.head_num+1)//2, segment_size+1, step=self.head_num, device=inputs_embeds.device).unsqueeze(0)
        # [1,seq_len+mem_size]
        encode_position_ids = torch.cat([position_ids, mem_position_ids],dim=1)

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
                    enabled_lora=False
                )

        hidden_states = outputs.hidden_states[-1]

        # [B,mem_size,emb_size]
        mem_hidden = hidden_states[:,-mem_size:]
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

        mask.update({"cl_prime_mask": cl_prime_mask, "lm_mask": lm_mask,})
    else:
        cl_zero_mask = torch.zeros_like(input_token, dtype=torch.bfloat16).to(input_token.device)
        cl_ones_mask = torch.ones_like(compress_token, dtype=torch.bfloat16).to(input_token.device)
        cl_mask = torch.cat([cl_zero_mask, cl_ones_mask], dim=1).to(input_token.device)

        lm_ones_mask = torch.ones_like(input_token, dtype=torch.bfloat16).to(input_token.device)
        lm_zero_mask = torch.zeros_like(compress_token, dtype=torch.bfloat16).to(input_token.device)
        lm_mask = torch.cat([lm_ones_mask, lm_zero_mask], dim=1).to(input_token.device)

        mask.update({"cl_mask": cl_mask, "lm_mask": lm_mask,})
    return mask



def save_adapter(model,save_path_and_name='adapter.pt', log=False):
    adapter_name = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if log:
                print("[Save Adapter]",name)
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
        for name, module in model.named_children():     # adapter是W'=W+AB -> instruction_adapter是
            if name == "compress_head":
                continue
            if isinstance(module, LinearLoraLayer) or isinstance(module, EmbeddingLoraLayer):
                lora_AB = module.lora_A.data @ module.lora_B.data
                if module.weight.data.shape == lora_AB.shape:
                    module.weight.data += lora_AB * module.scale
                else:
                    module.weight.data += lora_AB.transpose(0,1) * module.scale
            else:
                merge_weight(module)

    def init_lora(model, task_config):
        for name, module in model.named_children():
            if name == "compress_head":
                continue
            if isinstance(module, LinearLoraLayer):
                setattr(model, name,
                        LinearLoraLayer(module.in_features, module.out_features, r=16, weight=module.weight.data.clone()))
            elif isinstance(module, EmbeddingLoraLayer):
                setattr(model, name,
                        EmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx, r=128, weight=module.weight.data.clone()))
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
                        LinearLoraLayer(module.in_features, module.out_features, r=16, weight=module.weight.data.clone()))
            elif isinstance(module, nn.Embedding):
                setattr(model, name, EmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx, r=128,
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
                        TripleLinearLoraLayer(module.in_features, module.out_features, r_cl=16, r_lm=16, r_cl_prime=16, weight=module.weight.data.clone()))
            elif isinstance(module, nn.Embedding):
                setattr(model, name, TripleEmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx,
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
        modify_llama()
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
# python /home/liuxinyu/zrs/forget-me-not/models/llama3.py