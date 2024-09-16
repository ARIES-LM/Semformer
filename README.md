# Semformer

![image](https://github.com/ARIES-LM/Semformer/blob/master/model-1.png)

Next-token prediction serves as the dominant component in current neural language models. During the training phase, the model employs teacher forcing, which involves predicting tokens based on all preceding ground truth tokens. However, this approach has been found to create shortcuts, utilizing the revealed prefix to spuriously fit future tokens, potentially compromising the accuracy of the next-token predictor. We introduce Semformer, a Transformer language model explicitly modeling the semantic planning of response. Specifically, we incorporate a sequence of planning tokens into the prefix, guiding the planning token representations to predict the latent semantic representations of the response, which are induced by an autoencoder. In a minimal planning task (graph path-finding), Semformer exhibits near-perfect performance, effectively mitigating shortcut learning-a feat that standard training and baselines fail to achieve. Furthermore, we pretrain Semformer from scratch with 125M parameters, demonstrating its efficacy through measures of perplexity, in-context learning, and fine-tuning on summarization tasks.

## Requirement
transformers==4.40

torch>=2.0+cu121

## Next-Token-Failures

Bachmann et al., 2024 present a Clever Hans Cheat phenomenon characterized by shortcut learning in teacher forcing on a minimal planning task (graph path-finding). We use it as the main testbed.

### Generate training and test data:

cd data

\# Change length of path and degree

python3 graphs.py

### Train and evaluate model: 

bash finetune_semformer.sh

More details refer to the original repository (https://github.com/gregorbachmann/Next-Token-Failures)

## LM pretraining

We also use transformers to train a GPT2-125M on OpenwebText. The setting follows Backpack LM (https://aclanthology.org/2023.acl-long.506/).

```
modelname=xxx
accelerate launch \
    --num_processes 8 \
    --num_machines $ARNOLD_WORKER_NUM \
    --machine_rank $ARNOLD_ID \
    --main_process_ip $ARNOLD_WORKER_0_HOST \
    --main_process_port $ARNOLD_WORKER_0_PORT \
    --use_deepspeed \
    --zero_stage 1 \
    --deepspeed_multinode_launcher standard \
    run_clm.py \
    --dataset_name openwebtext \
    --model_name_or_path gpt2 --preprocessing_num_workers 128 \
    --load_pretrained_enc_and_fix 1 --ztokens 16 --encstr suffix \
    --alpha 0.5 --beta 1 --zdim 64 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 --logging_steps 50 \
    --weight_decay 0.1 --learning_rate 6e-4 --lr_scheduler_type cosine --warmup_steps 5000 \
    --seed 42 --bf16 --block_size 1024 \
    --save_total_limit 1 --save_strategy steps --save_steps 5000 --do_train --max_steps 100000 \
    --output_dir $write_path/checkpoints/$modelname >>$write_path/logs/log.$modelname 2>&1
```


### Please kindly cite our paper if you find it helpful:
```
@Article{yin2024lexmatcher,
  author  = {Yongjing Yin, Junran Ding, Kai Song, Yue Zhang},
  title   = {Semformer: Transformer Language Models with Semantic Planning},
  journal = {},
  year    = {2024}
}
```

