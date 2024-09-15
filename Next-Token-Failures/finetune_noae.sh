
expath=.

export TRANSFORMERS_CACHE=${expath}/hfcache
export HF_HOME=${expath}/hfcache
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8

write_path=${expath}/pitfall_result

ngpu=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

modelname="l" # TODO: in ['mini', 's', 'm', 'l']
# modelname="s" # TODO: in ['mini', 's', 'm', 'l']

lr=1e-5
ep=100

nn=50 # num_nodes

deg=20
path=10

a=1
b=0

snl=6

ztokens=4

zdim=1280

if  [ $((deg * path)) -gt 50 ];
then
    nn=$((deg * path))
fi

if  [ "$modelname" = "s" -o "$modelname" = "mini" ];
then
    modelpath=hfmodels/gpt2
    bs=16
fi

if  [ "$modelname" = "mini" ];
then
    ep=500
    lr=5e-4
fi


if  [ "$modelname" = "m" ];
then
    modelpath='gpt2-medium'
    bs=32
fi

if  [ "$modelname" = "l" ];
then
    modelpath=hfmodels/gpt-large
    bs=2
fi

echo $modelpath
echo $bs
echo $nn
echo $lr

emamom=0.999
logname=ema${emamom}-$modelname-d${deg}p$path-z${ztokens}dim${zdim}a${a}-lr$lr

accelerate launch \
    --num_processes $(($ARNOLD_WORKER_NUM * $ARNOLD_WORKER_GPU)) \
    --num_machines $ARNOLD_WORKER_NUM \
    --machine_rank $ARNOLD_ID \
    --main_process_ip $ARNOLD_WORKER_0_HOST \
    --main_process_port $ARNOLD_WORKER_0_PORT \
    --mixed_precision bf16 \
    finetune.py \
    --model $modelpath \
    --use_flash_attention \
    --n_train 200000 \
    --n_test 20000 \
    --batch_size $bs \
    --no_ae \
    --use_ema $emamom \
    --epochs $ep \
    --eval_every 5000 \
    --dataset graph \
    --deg $deg \
    --path $path \
    --a $a \
    --b $b \
    --zdim $zdim \
    --ztokens $ztokens \
    --snl $snl \
    --num_nodes $nn \
    --save_every 500000 \
    --lr $lr \
    >>$write_path/logs/log.$logname 2>&1

