import math
import os
import warnings
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import random


import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.pytorch_utils import Conv1D
from transformers.modeling_outputs import (
    CausalLMOutput,
)

from transformers import (
    AutoModel,
    PreTrainedModel, 
    GPT2PreTrainedModel, 
    AutoModelForCausalLM
)

from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
logger = logging.get_logger(__name__)


class GPTNeoXAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them"
            )
        self.head_size = self.hidden_size // self.num_attention_heads
        self._init_bias(config.max_position_embeddings)

        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)

        self.norm_factor = self.head_size**-0.5
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.is_causal = True

    def _init_bias(self, max_positions, device=None):
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        if device is not None:
            self.bias = self.bias.to(device)


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        # Apply attention-specific projections and rope
        query, key, value, present = self._attn_projections_and_rope(
            hidden_states=hidden_states, position_ids=position_ids, layer_past=layer_past, use_cache=use_cache
        )

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn_projections_and_rope(
        self,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)


        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        
        present = (key, value) if use_cache else None

        return query, key, value, present

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=self.norm_factor,
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings

        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = True
        self.is_cross_attention = is_cross_attention
        self.reorder_and_upcast_attn = False

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)


        self.attn_dropout = nn.Identity()
        self.resid_dropout = nn.Identity()
        
        if hasattr(config, "attn_pdrop"):
            self.attn_dropout = nn.Dropout(config.attn_pdrop)
        
        if hasattr(config, "resid_pdrop"):
            self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        self.is_causal = True


    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with torch.amp.autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


@dataclass
class AEOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_z : torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PrefixEncoder(nn.Module):
    def __init__(self, config, model_args):
        super().__init__()

        if config.architectures[0] == "GPT2LMHeadModel":
            hidden_size = config.n_embd
            n_head = config.n_head
        elif config.architectures[0] == "GPTNeoXForCausalLM":
            hidden_size = config.hidden_size
            n_head = config.num_attention_heads

        self.input_dim = model_args.zdim

        self.hidden_dim = hidden_size

        self.prefix_seq_len = model_args.ztokens
        self.match_n_layer = model_args.shallow_decoder_n_layer
        
        self.prefix_mlp = nn.Linear(
            self.input_dim, self.match_n_layer * 2 * hidden_size)

        self.match_n_head = n_head
        self.match_n_embd = hidden_size // n_head

    def forward(
        self,
        input_embd
    ):
        
        batch_size = input_embd.size(0)
        past_key_values = self.prefix_mlp(input_embd)
        past_key_values = past_key_values.view(batch_size, self.prefix_seq_len, self.match_n_layer, -1)

        # Resize
        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_seq_len,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        
        # Transpose -> [match_n_layer*2, batch_size, match_n_head, prefix_seq_len, match_n_embd]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])
        past_key_values = torch.split(past_key_values, 2)
        
        all_kvs = ()
        for i in range(len(past_key_values)):
            kvpair = (past_key_values[i][0], past_key_values[i][1])
            all_kvs += (kvpair,)

        return all_kvs



class AE(PreTrainedModel):
    _supports_flash_attn_2 = True

    def __init__(self, config, model_args, lora_config=None):
        super().__init__(config)
        self.shallow_decoder_config = deepcopy(config)

        self.use_ztokens = config.task_ztokens if hasattr(config, "task_ztokens") else config.ztokens
        if config.architectures[0] == "GPT2LMHeadModel":
            hidden_size = config.n_embd
            self.shallow_decoder_config.n_layer = model_args.shallow_decoder_n_layer
            layer_norm_eps = config.layer_norm_epsilon

        elif config.architectures[0] == "GPTNeoXForCausalLM":
            hidden_size = config.hidden_size
            self.shallow_decoder_config.num_hidden_layers = model_args.shallow_decoder_n_layer
            layer_norm_eps = config.layer_norm_eps

        self.zwte = nn.Embedding(config.ztokens, hidden_size)

        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=self.shallow_decoder_config,
        )
        self.decoder.resize_token_embeddings(config.len_tokenizer)

        if lora_config is not None:
            self.encoder = get_peft_model(self.encoder, lora_config)

        self.cross_attention = GPT2Attention(config, is_cross_attention=True)

        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.prefix_encoder = PrefixEncoder(config, model_args)

        self.proj = None
        if hidden_size > config.zdim:
            self.proj = nn.Linear(hidden_size, config.zdim, bias=False)
        
        self.z_start_id = config.z_start_id
        self.z_end_id = self.z_start_id + self.use_ztokens
        
        self.init_newmodule()

    def init_newmodule(self):
        self.ln_1.bias.data.zero_()
        self.ln_1.weight.data.fill_(1.0)
        self.ln_2.bias.data.zero_()
        self.ln_2.weight.data.fill_(1.0)

        std = self.config.initializer_range
        self.zwte.weight.data.normal_(mean=0.0, std=std)

        if self.proj is not None:
            self.proj.weight.data.normal_(mean=0.0, std=std)

        for module in self.cross_attention.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()


    def tie_ae_encoder_main_decoder(self, main_decoder):
        if self.config.architectures[0] == "GPT2LMHeadModel":
            self.encoder = main_decoder.transformer
        elif self.config.architectures[0] == "GPTNeoXForCausalLM":
            self.encoder = main_decoder.gpt_neox

    def forward(
        self,
        input_ids_enc,
        input_ids_dec,
        labels,
        attention_mask_enc=None,
        attention_mask_dec=None,
    ):
        with torch.no_grad():
            enc_outs = self.encoder(
            input_ids=input_ids_enc,
            attention_mask=attention_mask_enc
            )
        enc_lhs = enc_outs.last_hidden_state

        bz = input_ids_enc.size(0)
        z_idx = torch.arange(0, self.use_ztokens, dtype=torch.long, device=input_ids_enc.device)

        input_ids_enc_z = z_idx.unsqueeze(0).repeat(bz, 1)
        hidden_states_z = self.zwte(input_ids_enc_z)


        residual = hidden_states_z
        hidden_states_z = self.ln_1(hidden_states_z)
        cross_outs = self.cross_attention(
            hidden_states = hidden_states_z,
            encoder_hidden_states = enc_lhs
        )
        hidden_z = cross_outs[0] + residual

        hidden_z = self.ln_2(hidden_z)

        if self.proj is not None:
            hidden_z = self.proj(hidden_z)        

        loss = None
        if labels is not None:
            past_key_values = self.prefix_encoder(hidden_z)
            if attention_mask_dec is not None:
                attention_mask_z = attention_mask_dec.new_ones(bz, hidden_z.size(1))
                attention_mask_cat = torch.cat(
                    (attention_mask_z, attention_mask_dec), dim=-1)
            else:
                attention_mask_cat = None

            dec_outs = self.decoder(
                input_ids=input_ids_dec,
                past_key_values=past_key_values,
                attention_mask=attention_mask_cat,
                output_hidden_states=True,
                output_attentions=False,
                labels=labels
            )
            loss = dec_outs.loss


        return AEOutput(
            loss=loss,
            hidden_z=hidden_z
        )


class Semformer(PreTrainedModel):
    _supports_flash_attn_2 = True

    def __init__(self, config, model_args):
        print("*******************************")
        print(config)

        if config.architectures[0] == "GPT2LMHeadModel":
            hidden_size = config.n_embd
        elif config.architectures[0] == "GPTNeoXForCausalLM":
            hidden_size = config.hidden_size

        if model_args.zdim == -1:
            model_args.zdim = hidden_size
        
        print(model_args)
        
        super().__init__(config)

        self.ztokens = model_args.ztokens
        
        self.alpha = model_args.alpha
        self.beta = model_args.beta
        
        self.model_args = model_args
        
        self.mseloss = F.smooth_l1_loss
        #self.mseloss = nn.MSELoss()

        self.main_decoder = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                config=config,
            )
        self.resize_token_embeddings(config.len_tokenizer)
        
        if self.alpha > 0:
            self.aemodel = AE(deepcopy(config), model_args)
        
        self.softmax = nn.Softmax(dim=-1)
        self.use_bowloss = model_args.use_bowloss

        self.proj = None
        if self.use_bowloss == 0 and model_args.zdim != hidden_size:
            self.proj  = nn.Linear(hidden_size, model_args.zdim, bias=False)
        
        self.bceloss = BCEWithLogitsLoss(reduction='sum')

        
    def build_ed(self, len_tokenizer):
        if self.alpha > 0:
            print("share ae encoder and main decoder")
            self.aemodel.tie_ae_encoder_main_decoder(self.main_decoder)

    def resize_token_embeddings(self, len_t):
        self.main_decoder.resize_token_embeddings(len_t)
        
    def generate(self, *args, **kwargs):
        return self.main_decoder.generate(*args, **kwargs)
    
    def forward(
        self,
        input_ids, 
        labels, 
        input_ids_enc,
        labels_enc,
        input_ids_enc_z,
        pos_mask, 
        **kwargs
    ):
        bs = input_ids.size(0)
        
        main_dec_outs = self.main_decoder(
            input_ids = input_ids, 
            labels = labels,
            output_hidden_states = True,
            output_attentions = True
        )

        main_dec_lhs = main_dec_outs.hidden_states[-1] # bs seqlen h
        #'CausalLMOutputWithCrossAttentions' object has no attribute 'last_hidden_state' 
        bs, seqlen, hz = main_dec_lhs.size()

        main_hidden_z = main_dec_lhs[pos_mask]

        if self.proj is not None:
            main_hidden_z = self.proj(main_hidden_z)
        
        mseloss = 0
        recloss = 0
        bowloss = 0
        
        if self.alpha > 0:
            ae_outs = self.aemodel(
                input_ids_enc=input_ids_enc,
                input_ids_dec=input_ids_enc,
                labels=labels_enc
            )

            mseloss = self.mseloss(main_hidden_z.float(), ae_outs.hidden_z.reshape(-1, ae_outs.hidden_z.size(-1)).detach().float())
            recloss = ae_outs.loss
        
        nllloss = main_dec_outs.loss
        
        tloss = self.alpha * mseloss  \
            + self.beta * recloss  \
            + nllloss 
            
        # calculating bowloss
        if self.use_bowloss > 0:
            mhz = main_dec_lhs[pos_mask]
            mhz = mhz.reshape(bs, -1, hz)
            mhz = mhz.mean(dim=1) # B H

            logits_bow = self.main_decoder.lm_head(mhz)
            
            vocab_size = self.main_decoder.vocab_size
            
            bow_target = labels[labels!=-100].reshape(bs,-1)
            bow_target = F.one_hot(bow_target, num_classes = vocab_size)
            bow_target = torch.sum(bow_target, dim=1).type(torch.bool).float()
            
            bowloss = self.bceloss(logits_bow, bow_target) / bs
            tloss += self.use_bowloss * bowloss

        self.output_results(mseloss, recloss, nllloss, bowloss)
        
        preds = main_dec_outs.logits.argmax(dim=-1)
        preds = preds[:, :-1]
        labels = labels[:,1:]
        acc = self.accuracy(preds, labels)
        
        return main_dec_outs.logits, tloss if self.training else nllloss, acc

    def accuracy(self, preds, labels):
        bz = labels.size(0)
        #print(labels)
        labels = labels[labels!=-100].reshape(bz, -1)
        #print(labels)
        preds = preds[:, -labels.size(1):]
        #print(preds)
        #print(preds.shape, labels.shape)

        correct = preds.eq(labels).to(torch.float)
        seq_correct = torch.sum(correct, dim=1).eq(labels.size(1)).float()
        acc = torch.mean(seq_correct)
        per_token_acc = correct.mean(dim=0)
        return {
            'acc' : acc,
            'token_acc' : per_token_acc
        }
    
    def output_results(self, mseloss, recloss, nllloss, bowloss, mode='p'):
        log = (
            f"{self.training}; "
            f"mseloss = {self.alpha} * {mseloss:.6f}, "
            f"recloss = {self.beta} * {recloss:.6f}, "
            f"nllloss = {nllloss:.6f}, "
        )
        if self.use_bowloss > 0:
            log += f"bowloss = {self.use_bowloss} * {bowloss:.6f}"
        log += '\n'
        
        if mode == 'w':
            with open(f'./nnn530_bll/{self.model_args.spname}.txt', 'a') as f:
                f.write(log)
        elif mode == 'p':
            print(log)
        else:
            return
