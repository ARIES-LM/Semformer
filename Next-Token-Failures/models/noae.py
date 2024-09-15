from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union


import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from contextlib import nullcontext
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

class DummyLinear:
    def __init__(self):
        pass
    def __call__(self, x):
        return x



class GPT2MLP(nn.Module):
    def __init__(self, config, embed_dim, output_dim, ind_dim=None):
        super().__init__()
        
        if ind_dim is None:
            ind_dim = embed_dim
        
        self.c_fc = nn.Linear(embed_dim, ind_dim)
        self.c_proj = nn.Linear(ind_dim, output_dim)
        self.act = ACT2FN[config.activation_function]

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states



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
        
        self.mseloss = F.smooth_l1_loss if model_args.msenorm == 1 else nn.MSELoss()

        self.main_decoder = GPT2LMHeadModel(self.config)
        
        self.softmax = nn.Softmax(dim=-1)
        hidden_size = config.hidden_size

        if model_args.predictor == "mlp":
            self.proj1 = GPT2MLP(config, config.hidden_size, model_args.zdim)
        else:
            self.proj1  = nn.Linear(hidden_size, model_args.zdim, bias=False)
        
        self.znorm_func = nn.LayerNorm(model_args.zdim, eps=config.layer_norm_epsilon, elementwise_affine=False)
        
        assert not model_args.use_separate

        if model_args.zdim != hidden_size:
            if model_args.predictor == "mlp":
                self.projm = GPT2MLP(config, config.hidden_size, model_args.zdim)
            else:
                self.projm  = nn.Linear(hidden_size, model_args.zdim, bias=False)
        else:
            self.projm  = DummyLinear()

        self.m = model_args.use_ema
        
        self.context = torch.no_grad()
    
    def build_ed(self, len_tokenizer):
        if not self.model_args.from_scratch:
            print("load pretrained", self.model_args.model_name_or_path)
            self.main_decoder = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                config=self.config
            )
        self.resize_token_embeddings(len_tokenizer)
        
        if not self.model_args.use_ema:
            # tie encoder again
            print("tie main_decoder and encoder")
            self.encoder = self.main_decoder

            if not isinstance(self.projm, DummyLinear):
                self.projm.requires_grad_(False)
        else:
            print("copy main_decoder to encoder")
            self.encoder = deepcopy(self.main_decoder)

            if not isinstance(self.projm, DummyLinear):
                self.projm.requires_grad_(False)

            self.encoder.requires_grad_(False)
    
    @torch.no_grad()
    def _momentum_update_encoder(self):
        """
        Momentum update of the encoder
        """
        for param_enc, param_dec in zip(self.encoder.parameters(), self.main_decoder.parameters()):
            param_enc.data = param_enc.data * self.m + param_dec.data * (1.0 - self.m)

        if not isinstance(self.projm, DummyLinear):
            for param_enc, param_dec in zip(
                self.projm.parameters(), self.proj1.parameters()
                ):
                param_enc.data = param_enc.data * self.m + param_dec.data * (1.0 - self.m)

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
            output_attentions = False
        )
        main_dec_lhs = main_dec_outs.hidden_states[-1] # bs seqlen h
        #'CausalLMOutputWithCrossAttentions' object has no attribute 'last_hidden_state' 
        main_hidden_z = main_dec_lhs[pos_mask].view(bs, -1, main_dec_lhs.size(-1))
        main_hidden_z = self.proj1(main_hidden_z)

        if self.model_args.znorm:
            main_hidden_z = self.znorm_func(main_hidden_z)

        with self.context:
            second_pass = self.encoder(
                input_ids = torch.cat((input_ids_enc, input_ids_enc_z), dim=-1), 
                labels = None,
                output_hidden_states = True,
            )
            second_pass_hiddens = second_pass.hidden_states[-1][:, -self.ztokens:,:]

            second_pass_hiddens = self.projm(second_pass_hiddens)

            if self.model_args.znorm:
                second_pass_hiddens = self.znorm_func(second_pass_hiddens)
            second_pass_hiddens = second_pass_hiddens.reshape(-1,second_pass_hiddens.size(-1))

        #print(main_hidden_z.shape, ae_outs.hidden_z.shape)
        mseloss = self.mseloss(main_hidden_z.float(), second_pass_hiddens.float())

        nllloss = main_dec_outs.loss
        
        tloss = self.alpha * mseloss + nllloss

        print(f"{self.training}; mseloss = {self.alpha} * {mseloss:.6f}, nllloss = {nllloss:.6f}")

        preds = main_dec_outs.logits.argmax(dim=-1)
        
        preds = preds[:, :-1]
        labels = labels[:,1:]
        #print(f"preds: {preds[labels != -100]}")
        #print(f"golds: {labels[labels != -100]}")
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
        
