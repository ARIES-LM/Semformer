
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers import (
    AutoModel,
    PreTrainedModel,
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
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block

from peft import get_peft_model

from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.modeling_outputs import (
    CausalLMOutput,
)


def init_weight(m, config):
    for module in m.modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

@dataclass
class CausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reppred_loss: Optional[torch.FloatTensor] = None
    rec_loss: Optional[torch.FloatTensor] = None


@dataclass
class AEOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_z: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PrefixEncoder(nn.Module):
    def __init__(self, config, model_args):
        super().__init__()

        self.input_dim = config.zdim
        self.hidden_dim = config.n_embd

        self.prefix_seq_len = config.ztokens

        self.match_n_layer = model_args.shallow_decoder_n_layer

        self.prefix_generator = model_args.prefix_generator

        if model_args.prefix_generator == "mlp":
            self.prefix_mlp = nn.Linear(
                self.input_dim, self.match_n_layer * 2 * config.n_embd, False)
        else:
            self.scales = nn.Parameter(
                torch.ones([1]))
            
            num_prompts = self.prefix_seq_len
            self.prompts = nn.Parameter(
                torch.empty(
                [
                    2 * self.match_n_layer, 
                    num_prompts, config.n_embd
                ]))

            for i in range(self.prompts.shape[0]):
                nn.init.xavier_uniform_(self.prompts[i])

        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head

    def forward(
        self,
        input_embd
    ):

        batch_size = input_embd.size(0)
        
        if self.prefix_generator == "mlp":
            past_key_values = self.prefix_mlp(input_embd)

            past_key_values = past_key_values.view(
                batch_size, self.prefix_seq_len, self.match_n_layer, -1)

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

        else:
            past_key_values = []
            scale = torch.maximum(torch.ones([]), self.scales[0])

            for j in range(self.match_n_layer):
                k = self.prompts[2*j][None, :, :] * scale
                v = self.prompts[2*j+1][None, :, :] * scale
                # batch_size prefix_seq_len h
                k = k.repeat([batch_size, 1, 1]).view(batch_size, self.prefix_seq_len, self.match_n_head, self.match_n_embd)
                v = v.repeat([batch_size, 1, 1]).view(batch_size, self.prefix_seq_len, self.match_n_head, self.match_n_embd)
                k = k.permute([0, 2, 1, 3])
                v = v.permute([0, 2, 1, 3])

                past_key_values.append((k, v))

        all_kvs = ()
        for i in range(len(past_key_values)):
            kvpair = (past_key_values[i][0], past_key_values[i][1])
            all_kvs += (kvpair,)

        return all_kvs


class AE(PreTrainedModel):
    _supports_flash_attn_2 = True

    def __init__(self, config, model_args, lora_config=None):
        super().__init__(config)
        self.use_ztokens = config.ztokens
        
        self.zwte = nn.Embedding(config.ztokens, config.n_embd)
        
        self.encoder = GPT2Model.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            )
        
        dec_config = deepcopy(config)
        dec_config.n_layer = model_args.shallow_decoder_n_layer
        dec_config.max_position_embeddings = config.max_position_embeddings + config.ztokens

        if model_args.from_scratch:
            self.decoder = GPT2LMHeadModel(dec_config)
        else:
            self.decoder = GPT2LMHeadModel.from_pretrained(
                model_args.model_name_or_path,
                config=dec_config,
        )

        if lora_config is not None:
            self.encoder = get_peft_model(self.encoder, lora_config)

        block_config = deepcopy(self.config)

        block_config.add_cross_attention = True
        block_config._attn_implementation = "eager"

        self.cross_block = GPT2Block(block_config)

        self.lnz = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.prefix_encoder = PrefixEncoder(config, model_args)

        self.proj = None
        if config.hidden_size > config.zdim:
            self.proj = nn.Linear(config.hidden_size, config.zdim, bias=False)
        
        self.z_start_id = config.z_start_id
        self.z_end_id = self.z_start_id + self.use_ztokens

        init_weight(self.proj, config)
        init_weight(self.prefix_encoder, config)
        init_weight(self.lnz, config)
        init_weight(self.cross_block, config)
        init_weight(self.zwte, config)

    def forward(
        self,
        input_ids_enc,
        attention_mask_enc,
        input_ids_dec,
        attention_mask_dec,
        labels,
    ):
        with torch.no_grad():
            if isinstance(self.encoder, GPT2Model):
                enc_outs = self.encoder(
                input_ids=input_ids_enc,
                attention_mask=attention_mask_enc
                )
            else:
                enc_outs = self.encoder.transformer(
                input_ids=input_ids_enc,
                attention_mask=attention_mask_enc
                )


        bz = input_ids_enc.size(0)
        z_idx = torch.arange(0, self.use_ztokens, dtype=torch.long, device=input_ids_enc.device)

        input_ids_enc_z = z_idx.unsqueeze(0).repeat(bz, 1)
        hidden_states_z = self.zwte(input_ids_enc_z)

        attention_mask_enc_4d = attention_mask_enc.view(
            input_ids_enc.size(0), -1)
        attention_mask_enc_4d = attention_mask_enc_4d[:, None, None, :]

        attention_mask_enc_4d = attention_mask_enc_4d.to(dtype=self.dtype)
        attention_mask_enc_4d = (
            1.0 - attention_mask_enc_4d) * torch.finfo(self.dtype).min

        enc_lhs = enc_outs.last_hidden_state

        hidden_z = self.cross_block(
            hidden_states_z,
            encoder_attention_mask = attention_mask_enc_4d,
            encoder_hidden_states = enc_lhs
        )[0]

        hidden_down = self.lnz(hidden_z)
        
        if self.proj is not None:
            hidden_down = self.proj(hidden_down)        

        loss = None
        if labels is not None:
            past_key_values = self.prefix_encoder(hidden_down)
            attention_mask_z = attention_mask_dec.new_ones(bz, hidden_down.size(1))

            dec_outs = self.decoder(
                input_ids=input_ids_dec,
                past_key_values=past_key_values,
                attention_mask=torch.cat(
                    (attention_mask_z, attention_mask_dec), dim=-1),
                output_hidden_states=True,
                output_attentions=False,
                labels=labels
            )
            loss = dec_outs.loss

        return AEOutput(
            loss=loss,
            hidden_z=hidden_down
        )


class Semformer(PreTrainedModel):
    _supports_flash_attn_2 = True

    def __init__(self, config, model_args, lora_config=None):
        super().__init__(config)
        self.alpha = model_args.alpha
        self.beta = model_args.beta
        if self.beta > 0 or self.alpha > 0:
            self.aemodel = AE(deepcopy(config), model_args=model_args, lora_config=lora_config)
        
        self.use_ztokens = config.task_ztokens

        self.z_start_id = config.z_start_id
        self.z_end_id = self.z_start_id + self.use_ztokens

        self.model_args = model_args

        self.mseloss = nn.MSELoss()

        self.model_parallel = False
        self.device_map = None

        dec_config = deepcopy(config)
        dec_config.max_position_embeddings += config.task_ztokens
        
        if model_args.from_scratch:
            self.main_decoder = GPT2LMHeadModel(dec_config)
        else:
            self.main_decoder = GPT2LMHeadModel.from_pretrained(
                self.model_args.model_name_or_path,
                config=dec_config,
                ignore_mismatched_sizes=True
            )
        # will revise config.vocab_size
        self.main_decoder.resize_token_embeddings(config.len_tokenizer)

        self.proj = None
        if config.hidden_size > config.zdim:
            if self.use_ztokens != config.ztokens:
                self.proj = nn.Linear(config.hidden_size, config.zdim * config.ztokens, bias=False)
            else:
                self.proj = nn.Linear(config.hidden_size, config.zdim, bias=False)
            
            self.proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    
    def share_emb_decoders(self):
        if self.beta > 0 or self.alpha > 0:
            print(f"share emb lmhead between two decoders")
            self.aemodel.decoder.lm_head = self.main_decoder.lm_head
            self.aemodel.decoder.transformer.wte = self.main_decoder.transformer.wte


    def freeze_ae(self):
        if hasattr(self, "aemodel"):
            print(f"fix ae")
            self.aemodel.requires_grad_(False)


    def freeze_lm(self):
        print(f"fix lm")
        self.main_decoder.requires_grad_(False)


    def load_encoder_and_fix(self, model_name_or_path, config):
        if hasattr(self, "aemodel"):
            print(f"load pretrained and fix encoder")
            self.aemodel.encoder = GPT2Model.from_pretrained(
                model_name_or_path,
                config=config,
                )
            self.aemodel.encoder.requires_grad_(False)
    
    def tie_aeencoder_with_decoder(self):
        if hasattr(self, "aemodel"):
            print("/tie encoder/ with main decoder")
            self.aemodel.encoder = self.main_decoder

    def resize_token_embeddings(self, len_t):
        self.main_decoder.resize_token_embeddings(len_t)

    def generate(self, *args, **kwargs):
        return self.main_decoder.generate(*args, **kwargs)


    def forward(
        self,
        input_ids,
        input_ids_enc,
        input_ids_ae_dec,
        attention_mask,
        attention_mask_enc,
        attention_mask_ae_dec,
        labels,
        labels_ae,
        **kwargs
    ):

        bs = input_ids.size(0)

        if self.beta > 0 and self.alpha == 0:
            # pretrain ae
            ae_outs = self.aemodel(
                input_ids_enc=input_ids_enc,
                attention_mask_enc=attention_mask_enc,
                input_ids_dec=input_ids_ae_dec,
                attention_mask_dec=attention_mask_ae_dec,
                labels=labels_ae
            )
            recloss = ae_outs.loss

            return CausalLMOutput(
                        loss=recloss,
                        reppred_loss=torch.zeros_like(recloss),
                        rec_loss=recloss,
                        )


        main_dec_outs = self.main_decoder(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        nllloss = main_dec_outs.loss

        if self.alpha > 0:
            main_dec_lhs = main_dec_outs.hidden_states[-1]  # bs seqlen h
            # 'CausalLMOutputWithCrossAttentions' object has no attribute 'last_hidden_state'
            is_ztokens = self.z_start_id <= input_ids
            #  < self.z_end_id
            main_hidden_z = main_dec_lhs[is_ztokens].view(bs, -1, main_dec_lhs.size(-1))

            if self.beta == 0:
                ae_outs = self.aemodel(
                input_ids_enc=input_ids_enc,
                attention_mask_enc=attention_mask_enc,
                input_ids_dec=input_ids_ae_dec,
                attention_mask_dec=attention_mask_ae_dec,
                labels=None
                )
                recloss = torch.zeros_like(nllloss)
            else:
                ae_outs = self.aemodel(
                input_ids_enc=input_ids_enc,
                attention_mask_enc=attention_mask_enc,
                input_ids_dec=input_ids_ae_dec,
                attention_mask_dec=attention_mask_ae_dec,
                labels=labels_ae
                )
                recloss = ae_outs.loss

            target_z = ae_outs.hidden_z.detach()

            if self.use_ztokens != self.config.ztokens:
                # B *
                main_hidden_z = main_hidden_z[:, -1]
                target_z = target_z.view(bs, -1)

            if self.proj is not None:
                main_hidden_z = self.proj(main_hidden_z)

            mseloss = self.mseloss(main_hidden_z, target_z)

            tloss = self.alpha * mseloss \
                + self.beta * recloss \
                + nllloss

        else:
            tloss = nllloss
            mseloss = torch.zeros_like(tloss)
            recloss = torch.zeros_like(tloss)

        return CausalLMOutput(
            loss=tloss if self.training else nllloss,
            logits=main_dec_outs.logits,
            reppred_loss=self.alpha * mseloss,
            rec_loss=self.beta * recloss,
        )
