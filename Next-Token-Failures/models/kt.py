from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import random

from scipy.optimize import linear_sum_assignment

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
class KTOutput(ModelOutput):
 
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    acc: dict = None
    
    
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


        
        
class KT(GPT2LMHeadModel):
    def __init__(self, config, model_args):
        super().__init__(config)
        model_args.k = max(model_args.k, 1)
        self.model_args = model_args
        self.pree = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=self.config,
        )

        self.transformer = self.pree.transformer
        self.lm_head = self.pree.lm_head
        
        self.sub_heads = nn.ModuleList(
            [
                deepcopy(self.lm_head) #nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
                for _ in range(model_args.k-1)
            ]
        )
        print(self.config)
        
    def generate(self, *args,**kwargs):
        return self.pree.generate(*args,**kwargs)

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
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        #print("input_ids = ",input_ids)
        #print("labels = ", labels)
        
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        if self.model_args.k > 1 and self.training:
            sub_logits = ()
            for sub_head in self.sub_heads:
                sub_logits += (sub_head(hidden_states),)
        #print(lm_logits.argmax(dim=-1)[:, :-1])
        #if labels :print(labels[:,1:])
        loss_fct = CrossEntropyLoss()
        
        loss = None
        acc = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            block_size = labels.size(-1)
            if self.model_args.k > 1:
                # pad y in dim -1
                pad_labels = -100 * torch.ones(
                    (labels.size(0),self.model_args.k-1)
                ).type(torch.long).to(lm_logits.device)
                
                labels = torch.cat(
                    (labels, pad_labels), dim = -1
                )
            
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:block_size].contiguous()
            # Flatten the tokens
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            if self.model_args.k > 1 and self.training:
                for i in range(self.model_args.k-1):
                    sub_shift_logits = sub_logits[i][..., :-1, :].contiguous()
                    sub_shift_labels = labels[..., 2+i:block_size+i+1].contiguous()
                    loss += loss_fct(
                        sub_shift_logits.view(-1, shift_logits.size(-1)), 
                        sub_shift_labels.view(-1)
                    )
            
            loss = loss / self.model_args.k
            preds = lm_logits.argmax(dim=-1)[:, :-1]
            labels = labels[:,1:block_size]
            acc = self.accuracy(preds, labels)
            
        if labels is None:

            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )
        return KTOutput(
            loss=loss,
            logits=lm_logits,
            acc=acc,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        
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

    def get_all_child_modules(self,module):
        child_modules = list(module.children())
        all_child_modules = []
        for child in child_modules:
            all_child_modules.append(child)
            all_child_modules.extend(self.get_all_child_modules(child))

        return all_child_modules
    
    def r_init_weights(self, module):
        all_child_modules = self.get_all_child_modules(module)
        #print("all_child_modules are:", all_child_modules)
        for i in all_child_modules:
            self.s_init_weights(i)
            
    def s_init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        