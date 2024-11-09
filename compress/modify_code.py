import logging
import pdb

import math
import torch
import transformers
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from torch import nn
from transformers import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv



def LlamaForCausalLM__lora_forward(
        self,
        input_ids: torch.LongTensor = None,
        mask = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        mask = mask,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = 0
        for k,v in mask.items():
            if k == "cl_mask":
                hidden_states_cl = hidden_states * v
                logits_cl = [F.linear(hidden_states_cl, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits += logits_cl
            if k == "lm_mask":
                hidden_states_lm = hidden_states * v
                logits_lm = [F.linear(hidden_states_lm, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits += logits_lm
        logits = torch.cat(logits, dim=-1)
    else:
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :], mask)

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def LlamaModel_lora_forward(
        self,
        input_ids: torch.LongTensor = None,
        mask = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids, mask)

    past_seen_tokens = 0
    if use_cache:  # kept for BC (cache positions)
        if not isinstance(past_key_values, StaticCache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

    if cache_position is None:
        if isinstance(past_key_values, StaticCache):
            raise ValueError("cache_position is a required argument when using StaticCache.")
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                mask,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                mask=mask,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
        )
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def LlamaDecoderLayer_lora_forward(
    self,
    hidden_states: torch.Tensor,
    mask = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence
        position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
            with `head_dim` being the embedding dimension of each attention head.
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """
    # 用于后续残差连接
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    # 自注意力计算
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        mask=mask,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states, mask)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs



def LlamaSdpaAttention_lora_forward(
    self,
    hidden_states: torch.Tensor,
    mask = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            mask=mask,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states, mask)
    key_states = self.k_proj(hidden_states, mask)
    value_states = self.v_proj(hidden_states, mask)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output, mask)

    return attn_output, None, past_key_value


def LlamaMLP_lora_forward(self, x, mask): # x[batch_size, seq_len, hidden_size]
    if self.config.pretraining_tp > 1: # 假如pretraining_tp=2 张量并行切片数
        # todo: debug一下吧  看看维度
        print("-------------------------------------------------------------------切片----------------------------")
        slice = self.intermediate_size // self.config.pretraining_tp # 5504
        # weight[intermediate_size,hidden_size] ——> weight[11008,4096]
        # gate_proj_slices[slice,hidden_size] ——> gate_proj_slices[5504,4096]
        gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        # weight[hidden_size,intermediate_size] ——> weight[4096,11008]
        # down_proj_slices[hidden_size,slice] ——> [4096,5504]
        down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        # gate_proj：x[b,s,h] * gate_proj_slices[slice,h] ——> x[b,s,slice] 拼接 x[b,s,11048]
        gate_proj = torch.cat(
            [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        )
        up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

        # intermediate_states[b,s,i] ——> [b,s,slice]
        intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        # 输入[b,s,slice] ——> [b,s,h]([b,s,4096])
        down_proj = [
            F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
        ]
        down_proj = sum(down_proj)
    else:
        # x[b,s,h] ——> gate_proj(x)[b,s,i] ——> up_proj(x)[b,s,i] ——> down_proj()[b,s,h]
        repeat_mask = mask.copy()
        for k, v in mask.items():
            if k == "lm_mask" or k == "cl_mask" or k == "cl_prime_mask":
                repeat_mask[k] = v[:, :, :1]
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x, mask)) * self.up_proj(x, mask), repeat_mask)
    return down_proj

def repeat_masks(mask, old_dim, new_dim):
    if mask is None:
        return mask
    # repeat_factor: multiple of repetition
    repeat_factor = new_dim // old_dim
    input_len = mask["input_len"]
    compress_len = mask["compress_len"]
    new_masks = mask.copy()
    for k,v in mask.items():
        if k == "lm_mask":
            b, s, h = v.size()
            expanded_mask = v.repeat(1, 1, repeat_factor)  # 沿列方向重复
            # 计算余数部分的列数
            remaining_columns = new_dim - expanded_mask.size(2)
            # 为了让小余数部分也保持每行的 0 和 1 模式
            if remaining_columns > 0:
                padding = torch.cat([
                    torch.ones(b, input_len, remaining_columns, dtype=torch.bfloat16),  # 前 input_len 行填充 1
                    torch.zeros(b, compress_len, remaining_columns, dtype=torch.bfloat16),  # 后 compress_len 行填充 0
                ], dim=1).to(expanded_mask.device)
                # 拼接填充部分
                expanded_mask = torch.cat([expanded_mask, padding], dim=2)
            new_masks[k] = expanded_mask
        if k == "cl_mask" or k == "cl_prime_mask":
            b, s, h = v.size()
            expanded_mask = v.repeat(1, 1, repeat_factor)  # 沿列方向重复
            # 计算余数部分的列数
            remaining_columns = new_dim - expanded_mask.size(2)
            # 为了让小余数部分也保持每行的 0 和 1 模式
            if remaining_columns > 0:
                padding = torch.cat([
                    torch.zeros(b, input_len, remaining_columns, dtype=torch.bfloat16),  # 前 input_len 行填充 1
                    torch.ones(b, compress_len, remaining_columns, dtype=torch.bfloat16),  # 后 compress_len 行填充 0
                ], dim=1).to(expanded_mask.device)
                # 拼接填充部分
                expanded_mask = torch.cat([expanded_mask, padding], dim=2)
            new_masks[k] = expanded_mask
    return new_masks

def modify_llama():
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM__lora_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_lora_forward
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = LlamaDecoderLayer_lora_forward
    transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = LlamaSdpaAttention_lora_forward
    transformers.models.llama.modeling_llama.LlamaMLP.forward = LlamaMLP_lora_forward
