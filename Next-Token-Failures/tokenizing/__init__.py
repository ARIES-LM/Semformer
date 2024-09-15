import torch
from transformers import AutoTokenizer
from tokenizing.numeral_tokenizer import NumeralTokenizer


class Tokenizer:
    def __init__(self, encoder, decoder, vocab_size, name=None):
        self.encode = encoder
        self.decode = decoder
        self.vocab_size = vocab_size
        self.name = name

    def tokenize(self, data_list):
        """
        Takes a list of prefix-target pairs, tokenizes and concatenates them
        """
        out = []
        prefix_len = len(self.encode(data_list[0][0]))
        target_len = len(self.encode(data_list[0][1]))
        same_len = True

        for prefix, target in data_list:
            prefix = torch.tensor(self.encode(prefix))
            target = torch.tensor(self.encode(target))
            if not (len(prefix) == prefix_len and len(target) == target_len):
                same_len = False
            seq = torch.concatenate([prefix, target], dim=-1).long()
            out.append(seq)

        # Check if all prefixes and all targets have the same length
        if not same_len:
            print('Not all prefixes or targets have the same length!!')
        else:
            print('Equal sequence lengths!')

        return out, prefix_len, target_len

class ZTokenizer:
    def __init__(self, encoder, decoder, vocab_size, name=None, model_args=None, tokenizer=None):
        self.model_args = model_args
        self.tokenizer = tokenizer
        
        self.encode = encoder
        self.decode = decoder
        self.vocab_size = vocab_size
        self.name = name
        
    def tokenize(self, data_list):
        """
        Takes a list of prefix-target pairs, tokenizes and concatenates them
        """
        out = []
        prefix_len = len(self.encode(data_list[0][0]))
        target_len = len(self.encode(data_list[0][1]))
        same_len = True

        for prefix, target in data_list:
            prefix = torch.tensor(self.encode(prefix))
            target = torch.tensor(self.encode(target))
            if not (len(prefix) == prefix_len and len(target) == target_len):
                same_len = False
            seq = torch.concatenate([prefix, target], dim=-1).long()
            out.append(seq)

        # Check if all prefixes and all targets have the same length
        if not same_len:
            print('Not all prefixes or targets have the same length!!')
        else:
            print('Equal sequence lengths!')

        return out, prefix_len, target_len

def get_tokenizer(args, **kwargs):
    if args.model == 'gpt':
        t = NumeralTokenizer(args.num_nodes)
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=args.num_nodes + 4, name='numeral')
    elif args.model.startswith('gpt2'):
        t = AutoTokenizer.from_pretrained('gpt2')
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=50257 , name='gpt2')
    elif args.model.startswith('pythia'):
        t = AutoTokenizer.from_pretrained('EleutherAI/' + args.model)
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=50304, name='gpt2')
    elif args.model.startswith('phi'):
        t = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=51200, name='phi')
    else:
        t = AutoTokenizer.from_pretrained(args.model)
        model_args = kwargs.get("model_args")
        special_list = [f'<THO{idx}>' for idx in range(model_args.ztokens)]
        special_seq = ''.join(special_list)
        print(special_list)
        print(special_seq)
        t.add_special_tokens({'additional_special_tokens':special_list})
        print(t)
        tokenizer = ZTokenizer(
            encoder=t.encode, 
            decoder=t.decode, 
            vocab_size=len(t), 
            name='gpt2', 
            model_args=model_args, 
            tokenizer = t
        )
        
        tokenizer.zseq = None
        tokenizer.z_start_id = None
        if model_args.ztokens > 0:
            tokenizer.zseq = tokenizer.encode(special_seq)
            tokenizer.z_start_id = tokenizer.encode('<THO0>')[0]
            print("zseq:",tokenizer.zseq)
            print("z_start_id:",tokenizer.z_start_id)

    return tokenizer
