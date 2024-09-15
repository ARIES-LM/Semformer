import torch
from tqdm import tqdm

from utils.training_utils import AverageMeter


# Function to evaluate performance when generating
@torch.no_grad()
def evaluate(model, loader, ctx, temperature, top_k, results=None, mode='test', accelerator = None):
    """
    Generates sequences (without teacher-forcing) and calculates accuracies
    """
    num_prefix_tokens = loader.dataset.num_prefix_tokens
    num_target_tokens = loader.dataset.num_target_tokens

    if hasattr(loader.dataset.tokenizer, "model_args"):
        ztokens = loader.dataset.tokenizer.model_args.ztokens

    # Switch dataset and model to "eval" mode
    loader.dataset.eval()
    model.eval()
    total_acc = AverageMeter()
    tokens_corr = {i: AverageMeter() for i in range(num_target_tokens)}
    bar = tqdm(loader)

    #model.set_cache(loader.dataset.device)
    for x in bar:
        y = x[:, num_prefix_tokens + ztokens:].clone()
        x = x[:, :num_prefix_tokens + ztokens].clone()
        attn_mask = torch.ones_like(x)
        with ctx:
            if accelerator:
                gen_model = accelerator.unwrap_model(model)
            
            #y_pred = model.generate(x, num_target_tokens, temperature=temperature, top_k=top_k)
            y_pred = gen_model.generate(
                x, 
                attention_mask = attn_mask.to('cuda'),
                max_new_tokens = num_target_tokens, 
                min_new_tokens = num_target_tokens,
                temperature = temperature, 
                pad_token_id = 50256,
                top_k = top_k
            )

            y_pred = y_pred[:, -num_target_tokens:]
        
        #model.reset_cache()
        print("Num_tar_tokens = ",num_target_tokens )
        # print("Question = ", loader.dataset.tokenizer.decode(x[0]))
        # print("Pred = ",loader.dataset.tokenizer.decode(y_pred[0]))
        # print("Gold = ",loader.dataset.tokenizer.decode(y[0]))

        # Check how many tokens we get right and how many predictions are completely correct

        # added for multi-gpu
        all_predictions, all_targets = accelerator.gather_for_metrics((y_pred, y))
        correct = all_targets.eq(all_predictions).float()

        # correct = y.eq().float()

        # Completely correct
        completely_correct = torch.mean(correct.sum(dim=1).eq(num_target_tokens).to(torch.float))
        total_acc.update(completely_correct.item(), x.shape[0])

        # Individual token accuracy
        per_token_acc = correct.mean(dim=0)
        for i in range(num_target_tokens):
            tokens_corr[i].update(per_token_acc[i].item(), x.shape[0])

        # bar.set_description(f'{mode} accuracy: {total_acc.get(percentage=True):.2f}')

    #model.empty_cache()

    # Switch back to train mode
    loader.dataset.train()
    model.train()

    if results is not None:
        results[mode + '/accuracy'] = total_acc.get(percentage=True)
        for i in range(num_target_tokens):
            results[mode + '/token_' + str(i + 1)] = tokens_corr[i].get(percentage=True)
    return results


# Function to evaluate performance when applying teacher forcing
@torch.no_grad()
def evaluate_forced(model, loader, ctx, results=None, mode='test',  accelerator = None):
    """
    Generates sequences with teacher-forcing and calculates accuracies
    """
    num_target_tokens = loader.dataset.num_target_tokens
    total_acc, total_loss = AverageMeter(), AverageMeter()
    tokens_corr = {i: AverageMeter() for i in range(num_target_tokens)}
    bar = tqdm(loader)

    model.eval()
    
    for tp in bar:
        # Produce logits with teacher-forcing (i.e. like during training)
        with ctx:
            if isinstance(tp, list) or isinstance(tp, tuple):
                logits, loss, accs = model(*tp)
                bs = tp[0].shape[0]
            elif isinstance(tp, dict):
                ret = model(**tp)
                loss = ret.loss
                logits = ret.logits
                accs = ret.acc 
                bs = tp['input_ids'].shape[0]
            else:
                assert 0

        total_acc.update(val=accs['token_acc'], num=bs)
        total_loss.update(val=loss, num=bs)
        for i in range(num_target_tokens):
            tokens_corr[i].update(accs['token_acc'][i], bs)

        bar.set_description('Forced Loss: {:.4f} Forced Acc: {}'.format(total_loss.get(),
                                                              total_acc.get_tensor_for_display()))

    if results is not None:
        results[mode + '/forced loss'] = total_loss.get()
        results[mode + '/forced accuracy'] = total_acc.get(percentage=True)
        for i in range(num_target_tokens):
            results[mode + '/token_' + str(i + 1)] = tokens_corr[i].get(percentage=True)

    model.train()

    return results
