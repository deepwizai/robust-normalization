import copy
from tqdm import tqdm
import numpy as np
import wandb 

import torch
from torch.cuda.amp import GradScaler, autocast

from utils import *

def train_bert(net, 
                bert_model,
                criterion, 
                opti, 
                lr, 
                lr_scheduler, 
                train_loader, 
                val_loader, 
                epochs, 
                device, 
                iters_to_accumulate):

    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    iters = []
    train_losses = []
    val_losses = []

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):
            opti.zero_grad()
            
            # Converting to cuda tensors
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)

            # Obtaining the logits from the model
            prob = net(seq, attn_masks, token_type_ids)
            
            # Computing loss
            loss = criterion(prob, labels)
    
            loss.backward()
            opti.step()
            
        print(f"Epoch[{ep}/{epochs}] Training Loss :{loss}")
        wandb.log({'train-loss': loss})
            
        val_loss = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))
        wandb.log({'val-loss': val_loss})

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_ep = ep + 1

    # Saving the model
    os.makedirs("BERT-based-Ranking-for-Biomedical-Entity-Normalization/models", exist_ok=True)
    path_to_model='BERT-based-Ranking-for-Biomedical-Entity-Normalization/models/{}_lr_{}_val_loss_{}_ep_{}.pt'.format(bert_model.split('/')[-1], lr, round(best_loss, 5), best_ep)
    torch.save(net_copy.state_dict(), path_to_model)
    print("The model has been saved in {}".format(path_to_model))

    del loss
    torch.cuda.empty_cache()
    
    return path_to_model
