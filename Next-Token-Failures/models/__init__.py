from models.gpt import GPT
from models.pythia import Pythia
from models.config import GPTConfig


from transformers import (
    AutoConfig,

)



def get_model(args, **kwargs):
    if args.model == 'gpt':
        config = GPTConfig(n_layers=args.n_layer, n_heads=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                           bias=True, vocab_size=args.vocab_size, dropout=0, use_flash=args.use_flash,
                           teacherless_token=args.teacherless_token)
        model = GPT(config)

    elif args.model.startswith('gpt2'):
        model = GPT.from_pretrained(args.model, teacherless_token=args.teacherless_token)
        if args.block_size < 1024:
            model.crop_block_size(args.block_size)

    elif args.model.startswith('pythia'):
        model = Pythia.from_pretrained(args.model, teacherless_token=args.teacherless_token)
    else:
        
        model_args = kwargs.get("model_args")
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        # added
        tokenizer = kwargs.get("tokenizer")
        if args.use_minigpt:
            assert 'small' in args.model
            config.update(
                {
                    "n_layer" : 12,
                    "n_embd" : 384,
                    "n_head" : 6,
                }
            )
        config.update(
            {
                "ztokens" : model_args.ztokens,
                "zdim" : model_args.zdim,
                "z_start_id" : tokenizer.z_start_id,
                "len_tokenizer": tokenizer.vocab_size
            }
        )

        if model_args.use_flash_attention:
            config._attn_implementation = "flash_attention_2"


        if args.no_ae:
            print("not use autoencoder")

            if args.use_kt:
                print("use mutli-token prediction")
                from models.kt import KT as Semformer
            else:
                from models.noae import Semformer

        else:
            assert not args.use_kt
            if args.use_oldversion:
                print("use former version")
                from models.semformer_former import Semformer
            else:
                print("models.semformer")
                from models.semformer import Semformer
        
        tmodel = Semformer(config = config, model_args = model_args)

        if hasattr(tmodel, "build_ed"):
            tmodel.build_ed(tokenizer.vocab_size)
        
        # print(tmodel)

    return tmodel
