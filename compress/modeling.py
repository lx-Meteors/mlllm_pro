from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import math
import transformers

from modify_code import modify_llama


# from peft import prepare_model_for_kbit_training
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


class LinearLoraLayer(nn.Module):
    # No bias in LLama3 LinearLayer
    def __init__(self, in_features, out_features, r=16, weight=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = 2  # The alpha value is usually twice the rank
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = F.linear(x, self.weight)
        result += self.scale*(x@self.lora_A@self.lora_B)
        return result
    

class EmbeddingLoraLayer(nn.Module):
    # No bias in LLama3 LinearLayer
    def __init__(self, in_features, out_features, padding_idx, r=128, weight=None):
        super().__init__()
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = 2  # The alpha value is usually twice the rank
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B)
        
        
    def forward(self, x):
        result = F.embedding(x, self.weight, self.padding_idx)
        after_A = F.embedding(x, self.lora_A, self.padding_idx)
        result += self.scale*(after_A@self.lora_B)
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

        if "addition" in task_config and task_config["addition"] == "without_compress_loss":
            self.compress_head = None
        else:
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
        # todo:1.
        if self.task_config["use_multi_lora"]:
            mask = {"lm_mask": torch.ones_like(inputs['input_ids'])}
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"], mask)
        else:
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"])
        bsz, seq_len, emb_size = inputs_embeds.size()
        mem_size = self.mem_tokens.size(0)
        expand_mem = self.mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
        encode_inputs_embeds = torch.cat([inputs_embeds,expand_mem],dim=1)

        # [1,seq_len]
        position_ids = torch.arange(1,seq_len+1,device=inputs_embeds.device).unsqueeze(0)
        # [1,mem_size]：compress token position information, the step is compression ratio
        mem_position_ids = torch.arange((self.head_num+1)//2, seq_len+1, step=self.head_num, device=inputs_embeds.device).unsqueeze(0)
        # [1,seq_len+mem_size]
        encode_position_ids = torch.cat([position_ids,mem_position_ids],dim=1)

        # make three masks：cl_mask、lm_mask、cl_prime_mask
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
        mem_hidden = hidden_states[:,-mem_size:]
        
        
        tot_loss = 0
        tot_task = 0
        loss_info = {}



        # LM loss
        if 'lm_targets' in inputs and self.task_config["use_lm_loss"]:
            # print("lm_targets will be used")
            # [B,seq_len-1] -> [B,seq_len-1,E]
            # todo:3.
            if self.task_config["use_multi_lora"]:
                mask = {"lm_mask": torch.ones_like(inputs['lm_targets'][:, :-1])}
                lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'][:, :-1], mask)
            else:
                lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'][:, :-1])

            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)
            
            #                     [B,mem_size,E];     [B,1,E];      [B,seq_len-1,E]
            lm_emb = torch.cat([mem_hidden, expand_lm_token,lm_target_emb],dim=1)
            lm_position_ids = torch.cat([mem_position_ids,position_ids+seq_len-1],dim=1)
            # todo:4.
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
            logits = outputs.logits[:,mem_size:]
            logits = logits.contiguous().view(-1, self.vocab_size)
            inputs['lm_targets'] = inputs['lm_targets'].contiguous().view(-1).to(logits.device)

            lm_loss = self.loss_fct(logits, inputs['lm_targets'])
            loss_info["lm_loss"] = lm_loss.item()
            tot_loss += lm_loss
            tot_task += 1            




        # compress loss
        if "compress_targets" in inputs and self.compress_head is not None:
            # print("compress_targets will be used")
            # [B,mem_size,emb_size] -> [B,mem_size,head_num*vocab_size,]
            logits = self.compress_head(mem_hidden)
            logits = logits.float()
            logits = logits.contiguous().view(-1, self.vocab_size)
            inputs['compress_targets'] = inputs['compress_targets'].contiguous().view(-1).to(logits.device)

            compress_loss = self.loss_fct(logits, inputs['compress_targets'])
            loss_info["compress_loss"] = compress_loss.item()
            tot_loss += compress_loss
            tot_task += 1 


        use_ae = True
        if "wo_ae" in self.task_config and self.task_config["wo_ae"]:
            use_ae = False
        # AE loss
        if 'ae_targets' in inputs and use_ae:
            # print("ae_targets will be used")
            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_ae_token = self.special_tokens[0:1].unsqueeze(0).expand(bsz, 1, emb_size)
            
            #                     [B,mem_size,E];     [B,1,E];      [B,seq_len-1,E]
            ae_emb = torch.cat([mem_hidden, expand_ae_token, inputs_embeds[:,:-1,:]],dim=1)
            ae_position_ids = torch.cat([mem_position_ids,position_ids-1],dim=1)
            
            if "wo_pe" in self.task_config:
                outputs = self.model(
                inputs_embeds=ae_emb
            )
            else:
                outputs = self.model(
                position_ids=ae_position_ids,
                inputs_embeds=ae_emb
            )

            # [B,mem_size+S,V] -> [B,S,V]
            logits = outputs.logits[:,mem_size:]
            inputs['ae_targets'] = inputs['ae_targets'].contiguous().view(-1).to(logits.device)
            ae_loss = self.loss_fct(logits.contiguous().view(-1, self.vocab_size), inputs['ae_targets'])
            loss_info["ae_loss"] = ae_loss.item()
            tot_loss += ae_loss
            tot_task += 1      
        else:
            inputs['ae_targets'] = inputs['ae_targets'].contiguous().view(-1).to(logits.device)
            loss_info["ae_loss"] = -1

        loss = tot_loss/tot_task
        # return AE_logtis for validation.
        return {"loss":loss, "loss_info":loss_info, "logits":logits}

    def ae_inference(self,inputs,generate_num):
        # ->LlamaForCausalLM->LlamaModel->embed_tokens
        inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"])
        bsz, seq_len, emb_size = inputs_embeds.size()
        mem_size = self.mem_tokens.size(0)
        expand_mem = self.mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
        encode_inputs_embeds = torch.cat([inputs_embeds,expand_mem],dim=1)

        # [1,seq_len]
        position_ids = torch.arange(1,seq_len+1,device=inputs_embeds.device).unsqueeze(0)
        # [1,mem_size]
        mem_position_ids = torch.arange((self.head_num+1)//2, seq_len+1, step=self.head_num, device=inputs_embeds.device).unsqueeze(0)
        # [1,seq_len+mem_size]
        encode_position_ids = torch.cat([position_ids,mem_position_ids],dim=1)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        mem_hidden = hidden_states[:,-mem_size:]

        # [1,E] -> [1,1,E] -> [B,1,E]
        expand_ae_token = self.special_tokens[0:1].unsqueeze(0).expand(bsz, 1, emb_size)
            
        #                  [B,mem_size,E];   [B,1,E]
        ae_emb = torch.cat([mem_hidden, expand_ae_token],dim=1)
        ae_position_ids = torch.cat([mem_position_ids, position_ids[:,:1]-1],dim=1)
        

        generate_text = []
        past_key_values = None
        next_inputs_embeds = ae_emb.clone()
        next_position_ids = ae_position_ids.clone()
        
        for i in range(generate_num):
            
            if "wo_pe" in self.task_config:
                out = self.model(inputs_embeds=next_inputs_embeds, past_key_values=past_key_values, use_cache=True)
            else:
                out = self.model(position_ids=next_position_ids, inputs_embeds=next_inputs_embeds, past_key_values=past_key_values, use_cache=True)
            
            # [B,S,V] -> [B,V]
            logit = out.logits[:, -1]
            past_key_values = out.past_key_values
            # [B,V]->[B]
            next_token_id = torch.argmax(logit, dim=-1)

            # [B]->[B,E]->[B,1,E]
            next_inputs_embeds = self.model.model.embed_tokens(next_token_id).unsqueeze(1).to(inputs_embeds.device)
            next_position_ids = position_ids[:,i:i+1]
            generate_text.append(next_token_id.item())

        return generate_text


    def cl_inference(self, inputs, segment_size):
        # ->LlamaForCausalLM->LlamaModel->embed_tokens
        if self.task_config["use_multi_lora"]:
            mask = {"lm_mask": torch.ones_like(inputs['lm_targets'])}
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"], mask)
        else:
            inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"])
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
        for name, _ in adapter_state_dict.items():
            print(f"[Load Adapter] {name}")
    
    # 将adapter的权重转移到模型的设备上
    adapter_state_dict = {k: v.to(model.device) for k, v in adapter_state_dict.items()}
    
    model.load_state_dict(adapter_state_dict, strict=False)
    return model



def get_model_for_compress(model_id, task_config, rank):

    def add_compress_lora(model, task_config):
        for name, module in model.named_children():
            
            if name == "compress_head":
                continue
            if isinstance(module, nn.Linear):
                setattr(model, name, LinearLoraLayer(module.in_features, module.out_features, r=16, weight=module.weight.data.clone()))
            elif isinstance(module, nn.Embedding):
                setattr(model, name, EmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx, r=128, weight=module.weight.data.clone()))
            else:
                # Recursively apply this function to submodules
                add_compress_lora(module, task_config)

    def add_multi_lora(model, task_config):
        for name, module in model.named_children():
            if name == "compress_head":
                continue
            if isinstance(module, nn.Linear):
                setattr(model, name,
                        TripleLinearLoraLayer(module.in_features, module.out_features,  r_cl=16, r_lm=16, r_cl_prime=16, weight=module.weight.data.clone()))
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
    # todo:5.
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
# python /home/liuxinyu/zrs/forget-me-not/models/llama3.py