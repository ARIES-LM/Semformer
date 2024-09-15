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

from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.modeling_outputs import (
    CausalLMOutput,
)
@dataclass
class AEOutput(ModelOutput):
 
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_z : torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
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


class PrefixEncoder(nn.Module):
    def __init__(self, config, model_args):
        super().__init__()

        self.input_dim = model_args.zdim
        self.hidden_dim = config.n_embd

        self.prefix_seq_len = model_args.ztokens
        self.match_n_layer = model_args.shallow_decoder_n_layer
        
        self.prefix_mlp = nn.Linear(
            self.input_dim, self.match_n_layer * 2 * config.n_embd)


        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head

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


class AEEncoder(GPT2Model):
    def __init__(self, config, model_args):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.zwte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.zwpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        
    def disable_grad_except_z(self):
        self.requires_grad_(False)
        self.zwte.requires_grad_(True)
        self.zwpe.requires_grad_(True)
        
    def resume_main_decoder_grad(self):
        self.requires_grad_(True)
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        input_ids_z : Optional[torch.LongTensor] = None, # added
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        #self.disable_grad_except_z() # added
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        
        #torch.set_printoptions(profile='full', precision=5)
        #print(attention_mask)
        #assert 0, 'ckp1'
        
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds

        hidden_states = inputs_embeds

        # for z:        
        input_shape_z = input_ids_z.size()
        input_ids_z = input_ids_z.view(-1, input_shape_z[-1])
        position_ids_z = torch.arange(0, input_shape_z[-1], dtype=torch.long, device=device)
        position_ids_z = position_ids_z.unsqueeze(0).view(-1, input_shape_z[-1])
        inputs_embeds_z = self.zwte(input_ids_z)
        position_embeds_z = self.zwpe(position_ids_z)
        hidden_states_z = inputs_embeds_z + position_embeds_z
    
        hidden_states = torch.cat((hidden_states, hidden_states_z), dim=-2)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
        
        hidden_states = self.drop(hidden_states)

        output_shape = torch.Size((input_shape[0], input_shape[1] + input_shape_z[1]))\
                    + (hidden_states.size(-1),)
       
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        
        #self.resume_main_decoder_grad() # added
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class AEDecoder(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            
            attention_mask = attention_mask.view(batch_size, -1)
            
            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            #added
            with torch.no_grad():
                inputs_embeds = self.wte(input_ids)
        #added:
        with torch.no_grad():
            position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        
        
class AE(GPT2LMHeadModel):
    def __init__(self, config, model_args):
        super().__init__(config)
        self.ztokens = model_args.ztokens
        
        self.shallow_decoder_config = deepcopy(self.config)
        self.shallow_decoder_config.n_layer = model_args.shallow_decoder_n_layer
        
        self.decoder = AEDecoder(self.shallow_decoder_config)
        
        self.cross_attention = GPT2Attention(config, is_cross_attention=True)
        self.cross_attention.c_attn.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.cross_attention.q_attn.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln_1.bias.data.zero_()
        self.ln_1.weight.data.fill_(1.0)
        self.ln_2.bias.data.zero_()
        self.ln_2.weight.data.fill_(1.0)
        
        # self.prefix_encoder = LnPrefixEncoder(config, model_args)
        self.prefix_encoder = PrefixEncoder(config, model_args)
        
        self.proj = None
        if config.hidden_size > config.zdim:
            self.proj = nn.Linear(config.hidden_size, config.zdim, bias=False)
        
        self.post_init()
        
    def build_ed(self, model_args, main_decoder):
        if model_args.from_scratch:
            self.decoder = AEDecoder(self.shallow_decoder_config)
        else:
            self.decoder = AEDecoder.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=self.shallow_decoder_config
            )
        print(self.shallow_decoder_config)
        
        new_vocab_size = main_decoder.transformer.wte.weight.size(0)
        
        self.decoder.resize_token_embeddings(new_vocab_size)
        # self.transformer.resize_token_embeddings(new_vocab_size)
               
        self.decoder.wte = main_decoder.transformer.wte
        self.decoder.wpe = main_decoder.transformer.wpe
        self.lm_head = main_decoder.lm_head
        
        self.transformer = main_decoder.transformer
        
        self.zwte = nn.Embedding(self.transformer.config.vocab_size, self.transformer.embed_dim)
        self.zwpe = nn.Embedding(self.transformer.config.max_position_embeddings, self.transformer.embed_dim)
        self.zwte.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.zwpe.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        
    def forward(
        self,
        input_ids_enc,
        input_ids_enc_z,
        labels_enc,
    ):

        input_shape_z = input_ids_enc_z.size()
        input_ids_enc_z = input_ids_enc_z.view(-1, input_shape_z[-1])
        position_ids_z = torch.arange(0, input_shape_z[-1], dtype=torch.long, device=input_ids_enc_z.device)
        position_ids_z = position_ids_z.unsqueeze(0).view(-1, input_shape_z[-1])
        inputs_embeds_z = self.zwte(input_ids_enc_z)
        position_embeds_z = self.zwpe(position_ids_z)
        hidden_states_z = inputs_embeds_z + position_embeds_z
        

        with torch.no_grad():
            enc_outs = self.transformer(
                input_ids = input_ids_enc
            )
        
        enc_lhs = enc_outs.last_hidden_state 
        
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
        
        past_key_values = self.prefix_encoder(hidden_z)

        dec_outs = self.decoder(
            input_ids = input_ids_enc,
            past_key_values = past_key_values,
            output_hidden_states = True,
            output_attentions = False
        )

        lhs = dec_outs.last_hidden_state
        
        lm_logits = self.lm_head(lhs)
            
        loss = None
        if labels_enc is not None:
            # move labels to correct device to enable model parallelism
            labels_enc = labels_enc.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels_enc[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        rec_loss = loss
        return AEOutput(
            loss = rec_loss,
            hidden_z = hidden_z
        )
        
    
class Semformer(GPT2PreTrainedModel):
    def __init__(self, config, model_args):
        print("*******************************")
        print(config)
        if model_args.zdim == -1:
            model_args.zdim = config.hidden_size
        print(model_args)
        
        super().__init__(config)

        self.ztokens = model_args.ztokens
        
        self.alpha = model_args.alpha
        self.beta = model_args.beta
        
        self.model_args = model_args
        
        self.mseloss = F.smooth_l1_loss
        #self.mseloss = nn.MSELoss()
        
        self.main_decoder = GPT2LMHeadModel(self.config)
        
        if self.alpha > 0:
            self.aemodel = AE(deepcopy(config), model_args)
        
        self.softmax = nn.Softmax(dim=-1)
        self.use_bowloss = model_args.use_bowloss

        self.proj = None
        if self.use_bowloss == 0 and model_args.zdim != config.hidden_size:
            self.proj  = nn.Linear(config.hidden_size, model_args.zdim, bias=False)
        
        self.bceloss = BCEWithLogitsLoss(reduction='sum')
        
        
    def build_ed(self, len_tokenizer):
        if not self.model_args.from_scratch:
            self.main_decoder = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                config=self.config
            )
        self.resize_token_embeddings(len_tokenizer)

        if self.alpha>0:
            self.aemodel.build_ed(self.model_args, self.main_decoder)


    # DELLA
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.main_decoder.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.aemodel.transformer.h))
        self.main_decoder.parallelize(self.device_map)
        self.aemodel.transformer.parallelize(self.device_map)
        self.aemodel.decoder.parallelize(self.device_map)
        self.first_device = self.main_decoder.first_device
        self.model_parallel = True
        
    # DELLA
    def deparallelize(self):
        self.aemodel.encoder.deparallelize()
        self.aemodel.decoder.deparallelize()
        self.main_decoder.deparallelize()
        torch.cuda.empty_cache()
        
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
                input_ids_enc = input_ids_enc,
                input_ids_enc_z = input_ids_enc_z,
                labels_enc = labels_enc
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
