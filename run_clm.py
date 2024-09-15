#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
from collections import defaultdict
import random
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
from copy import deepcopy
import datasets
from datasets import load_dataset, Dataset, DatasetDict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.utils.versions import require_version

from transformers.trainer_utils import get_last_checkpoint, has_length, seed_worker
from transformers.utils import check_min_version, send_example_telemetry, is_datasets_available

from peft import get_peft_model,LoraConfig
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")

require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from aereg_gpt2_mem import Semformer
from trainer import MyTrainer as Trainer

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default='gpt2',
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    
    """
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )
    """

    # added
    ztokens: Optional[int] = field(default=32)
    zdim: Optional[int] = field(default=32)

    shallow_decoder_n_layer: Optional[int] = field(default=12)
    prefix_generator: Optional[str] = field(default='mlp')

    alpha: Optional[float] = field(default=1.0)
    beta: Optional[float] = field(default=1.0)

    from_scratch: Optional[bool] = field(default=True)

    use_flash_attention: Optional[bool] = field(default=True)
    use_lora: Optional[bool] = field(default=False)

    load_pretrained_enc_and_fix: Optional[int] = field(default=1)
    fix_aedecoder: Optional[int] = field(default=0)
    share_emb_decoders: Optional[int] = field(default=1)



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default='dummy.txt', metadata={
                                      "help": "The input training data file (a text file)."})

    validation_file: Optional[str] = field(
        default='dummy.txt',
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={
                            "help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    # added
    encstr: str = field(
        default='suffix',
        metadata={"help": "suffix or all"},
    )
    ptae: str = field(
        default=None,
        metadata={"help": "path or pretrained AE"},
    )

    from_disk: Optional[str] = field(default=None, metadata={"help": "load from disk."})
    already_tokened: Optional[bool] = field(default=False, metadata={"help": "load from disk."})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0",
                            "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def split_doc_by_sent(data_args, model_args, doc, tokenizer, special_seq):
    sentences = sent_tokenize(doc)
    sent_num = len(sentences)

    if sent_num < 3:
        return {}
    
    where_to_add = random.randint(1, sent_num-1)

    prefix = sentences[:where_to_add]
    suffix = sentences[where_to_add:]

    if data_args.encstr == "suffix":
        if data_args.append_z_aeenc:
            ae_encstr = ' '.join(suffix) + tokenizer.eos_token + special_seq
        else:
            ae_encstr = ' '.join(suffix) + tokenizer.eos_token
        
        ae_decstr = ' '.join(suffix) + tokenizer.eos_token
    elif data_args.encstr == "all":
        ae_encstr = doc + tokenizer.eos_token
        ae_decstr = doc + tokenizer.eos_token
    else:
        exit("encstr type")

    decstr = ' '.join(prefix) + special_seq + ' '.join(suffix) + tokenizer.eos_token

    return {"ae_encstr":ae_encstr, "ae_decstr":ae_decstr, "decstr":decstr}


def split_doc_(data_args, model_args, doc, tokenizer, special_seq):
    predict_len_min = 32
    
    doc_split = doc.split()
    
    if len(doc_split) < predict_len_min * 2 + model_args.ztokens:
        return {}

    doc_len = len(doc_split)
    
    # print("doc_len", doc_len)

    insert_posi = random.randint(predict_len_min, doc_len - predict_len_min - model_args.ztokens)
    
    prefix = doc_split[:insert_posi]
    suffix = doc_split[insert_posi:]

    if data_args.encstr == "suffix":
        if data_args.append_z_aeenc:
            ae_encstr = ' '.join(suffix) + tokenizer.eos_token + special_seq
            ae_decstr = special_seq + ' '.join(suffix) + tokenizer.eos_token
        else:
            ae_encstr = ' '.join(suffix) + tokenizer.eos_token
            ae_decstr = ' '.join(suffix) + tokenizer.eos_token

    elif data_args.encstr == "all":
        ae_encstr = doc + tokenizer.eos_token
        ae_decstr = doc + tokenizer.eos_token
    else:
        exit("encstr type")
    
    decstr = ' '.join(prefix) + special_seq + ' '.join(suffix) + tokenizer.eos_token
    
    return {"ae_encstr":ae_encstr, "ae_decstr":ae_decstr, "decstr":decstr}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_flash_attention:
        check_min_version("4.35.0")
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.

    special_list = [f'<THO{idx}>' for idx in range(model_args.ztokens)]
    special_seq = ''.join(special_list)
    tokenizer.add_special_tokens({'additional_special_tokens': special_list})

    tholist = [tokenizer.convert_tokens_to_ids(
                f'<THO{i}>') for i in range(model_args.ztokens)]

    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(tokenizer.pad_token_id)

    if data_args.from_disk is not None:
        raw_datasets = datasets.load_from_disk(data_args.from_disk)
    else:
        raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        streaming=data_args.streaming,
        )

    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)

    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base")
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)


    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        return output


    if data_args.already_tokened:
        logger.info("already_tokened")
        lm_datasets = raw_datasets
    else:
        with training_args.main_process_first(desc="dataset map tokenization and save to disk"):
            if not data_args.streaming:
                lm_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            else:
                lm_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        #     lm_datasets.save_to_disk(f"{data_args.dataset_name}_disk")
        # exit()

    predict_len_min = 32
    def group_texts(examples):
        # Concatenate all texts.
        result = defaultdict(list)

        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size

        for i in range(0, total_length, block_size):
            # insert tho tokens
            insert_posi = random.randint(predict_len_min, block_size - predict_len_min - model_args.ztokens)
            
            t = concatenated_examples["input_ids"]

            input_ids = t[i:i+insert_posi] + tholist + t[i+insert_posi:i+block_size]

            labels = t[i:i+insert_posi] + \
                            [-100] * len(tholist) + t[i+insert_posi:i+block_size]
            
            attention_mask = [1] * (len(labels))

            result["input_ids"].append(input_ids)
            result["attention_mask"].append(attention_mask)
            result["labels"].append(labels)

            # ae
            if data_args.encstr == "suffix":
                ae_enc_input_ids = t[i+insert_posi:i+block_size] + \
                    [tokenizer.pad_token_id] * (block_size-len(t[i+insert_posi:i+block_size]))
                ae_enc_attention_mask = [1]* (len(t[i+insert_posi:i+block_size])) + \
                                [0] * (block_size-len(t[i+insert_posi:i+block_size]))

                ae_dec_input_ids = t[i+insert_posi:i+block_size] + \
                        [tokenizer.pad_token_id] * (block_size-len(t[i+insert_posi:i+block_size]))
                ae_dec_attention_mask = [1]* (len(t[i+insert_posi:i+block_size])) + [0] * (block_size-len(t[i+insert_posi:i+block_size]))
                
                ae_dec_labels = ae_dec_input_ids.copy()
                ae_dec_labels = [-100 if ilabel in tholist or ilabel == tokenizer.pad_token_id else ilabel
                                                        for ilabel in ae_dec_labels]
            else:
                ae_enc_input_ids = t[i:i+block_size]
                ae_enc_attention_mask = [1]* (len(ae_enc_input_ids))

                ae_dec_input_ids = t[i:i+block_size]
                ae_dec_attention_mask = [1]* (len(ae_dec_input_ids))
                ae_dec_labels = ae_dec_input_ids.copy()

            result["input_ids_enc"].append(ae_enc_input_ids)
            result["attention_mask_enc"].append(ae_enc_attention_mask)

            result["input_ids_ae_dec"].append(ae_dec_input_ids)
            result["attention_mask_ae_dec"].append(ae_dec_attention_mask)
            result["labels_ae"].append(ae_dec_labels)

        return result


    with training_args.main_process_first(desc="group_doc insert tho tokens"):
        if not data_args.streaming:
            lm_datasets = lm_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="group_doc",
            )
        else:
            lm_datasets = lm_datasets.map(
                group_texts,
                batched=True,
            )
    logger.info(f"model args: {model_args}")

    z_start_id = tokenizer.convert_tokens_to_ids('<THO0>')
    config.update({"ztokens": model_args.ztokens, "zdim": model_args.zdim, "task_ztokens":model_args.ztokens,
                    "z_start_id":z_start_id, "len_tokenizer":len(tokenizer), "use_lora":model_args.use_lora})
    config._attn_implementation = "flash_attention_2"

    lora_config = None
    if model_args.use_lora:
        if config.architectures[0] == "GPT2LMHeadModel":
            # for gpt2
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["c_attn","c_proj","c_fc"],
                bias="none",
                task_type="CAUSAL_LM"
            )
        else:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                bias="none",
                task_type="CAUSAL_LM"
            )

    print(config)
    print("*************")
    if data_args.ptae is not None:
        logger.info(f"loading pretrained ae model {data_args.ptae}")
        tmodel = Semformer.from_pretrained(data_args.ptae, config=config, model_args=model_args, lora_config=lora_config)
    else:
        tmodel = Semformer(config=config, model_args=model_args, lora_config=lora_config)

    if model_args.fix_aedecoder:
        tmodel.freeze_aedecoer()

    if model_args.from_scratch:
        assert not model_args.use_lora

        if model_args.load_pretrained_enc_and_fix:
            tmodel.load_encoder_and_fix(model_args.model_name_or_path, config)
        else:
            tmodel.tie_aeencoder_with_decoder()
    
    if model_args.share_emb_decoders:
        tmodel.share_emb_decoders()

    if model_args.beta == 0:
        tmodel.freeze_ae()
    
    if model_args.alpha == 0:
        tmodel.freeze_lm()

    trainable_parameters = 0
    all_param = 0
    for pname, param in tmodel.named_parameters():
        all_param += param.numel()
        logger.info(f"{pname}, {param.requires_grad}")
        if param.requires_grad:
            trainable_parameters += param.numel()
    logger.info(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

    # Initialize our Trainer
    trainer = Trainer(
        model=tmodel,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        if not data_args.streaming:
            max_train_samples = (
               data_args.max_train_samples if data_args.max_train_samples is not None else len(
                   train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_train:
        kwargs = {"finetuned_from": model_args.model_name_or_path,
                  "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
