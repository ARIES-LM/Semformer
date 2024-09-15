expath=.

export TRANSFORMERS_CACHE=${expath}/hfcache
export HF_HOME=${expath}/hfcache
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8

write_path=${expath}/pitfall_result

ngpu=8

# gpu=0
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

modelname="l" # TODO: in ['mini', 's', 'm', 'l']

modelname="py-160" # TODO: in ['mini', 's', 'm', 'l']
modelname="py-410" # TODO: in ['mini', 's', 'm', 'l']

lr=1e-5
ep=100

nn=50 # num_nodes

deg=${1}
path=${2}

a=1.0
b=1.0

snl=6
ztokens=4
zdim=32


if  [ $((deg * path)) -gt 50 ];
then
    nn=$((deg * path))
fi

if  [ "$modelname" = "py-160" ];
then
    modelpath=hfmodels/pythia-160m
    bs=8
fi

if  [ "$modelname" = "py-410" ];
then
    modelpath=hfmodels/pythia-410m
    bs=4
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


logname=$modelname-d$deg-p$path-z$ztokens-a$a-b$b-zdim$zdim-lr$lr

accelerate launch \
    --num_processes $ngpu \
    --main_process_port $RANDOM \
    --mixed_precision bf16 \
    finetune.py \
    --model $modelpath \
    --use_flash_attention \
    --n_train 200000 \
    --n_test 20000 \
    --batch_size $bs \
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
    --lr $lr >>$write_path/logs/log.$logname 2>&1

