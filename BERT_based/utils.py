import os
import numpy as np
import random
from tqdm import tqdm 
import wandb 

import torch

def simple_tok(sent:str):
    return sent.split()

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def evaluate_loss(net, device, criterion, dataloader):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            prob = net(seq, attn_masks, token_type_ids)
            mean_loss += criterion(prob, labels).item()
            count += 1

    return mean_loss / count


def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy()

def test_prediction(net, device, dataloader, with_labels=True, result_file="results/output.csv", dataset_name='dataset name not provided'):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file
    """
    net.eval()
    w = open(result_file, 'w')
    w.write(f'{dataset_name} labels' + ',' + "probabilities" + '\n')
    probs_all = []
    labels_all = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, _ in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                prob = net(seq, attn_masks, token_type_ids)
                top_p, top_class = prob.topk(1, dim = 1)
                probs_all += top_p.view(-1).tolist()
                labels_all += top_class.view(-1).tolist()
        else:
            for seq, attn_masks, token_type_ids in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                prob = net(seq, attn_masks, token_type_ids)
                top_p, top_class = prob.topk(1, dim = 1)
                probs_all += top_p.view(-1).tolist()
                labels_all += top_class.view(-1).tolist()

    
    w.writelines(str(labels_all[i]) + ',' + str(probs_all[i]) + '\n' for i in range(len(probs_all)))
    w.close()
    
