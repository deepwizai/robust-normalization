import pandas as pd
import wandb 
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset

from utils import *
from data import dataset
from model import SentencePairClassifier
from train import train_bert
from evaluate import test_bert

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(
        df_train,
        df_val,
        df_test,
        bert_model,
        freeze_bert,
        maxlen,
        bs,
        device,
        lr,
        epochs,
        iters_to_accumulate,
        ):
    
    set_seed(1)

    
    print("Reading training data...")
    train_set = dataset(data=df_train, maxlen=maxlen, bert_model=bert_model)
    print("Reading validation data...")
    val_set = dataset(data=df_val, maxlen=maxlen, bert_model=bert_model)

    train_loader = DataLoader(train_set, batch_size=bs, num_workers=5)
    val_loader = DataLoader(val_set, batch_size=bs, num_workers=5)
        
    if freeze_bert==False:
        print("Finetuning Model...")
    net = SentencePairClassifier(bert_model=bert_model, freeze_bert=freeze_bert)
    net.to(device)
        
    y = torch.tensor(df_train['label'])
    class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y.numpy())
    print(f"class weights = {class_weights}")
    class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    
    # parameters = filter(lambda p: p.requires_grad, net.parameters())
    opti = AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    
    num_warmup_steps = 0 # The number of steps for the warmup phase.
    num_training_steps = epochs * len(train_loader)  # The total number of training steps
    t_total = (len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    path_to_model = train_bert(net=net, bert_model=bert_model, criterion=criterion, opti=opti,\
                               lr=lr, lr_scheduler=lr_scheduler, train_loader=train_loader,\
                               val_loader=val_loader, epochs=epochs, device=device, iters_to_accumulate=iters_to_accumulate)
    test_accuracy, test_f1_score = test_bert(bert_model=bert_model, 
                                    path_to_model=path_to_model, 
                                    path_to_output_file='pred_labels.txt', 
                                    df_test=df_test, bs=64, device=device)
    wandb.log({'test_accuarcy': test_accuracy,
                'test_f1_score': test_f1_score})

if __name__ == "__main__":
    # train, test and val sets are different for chemical set and for disease set. Please add paths accordingly
    df_train = pd.read_csv("train-data.csv")
    df_val = pd.read_csv("val-data.csv.gz")
    df_test = pd.read_csv("test-data.csv")
    
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_val = df_val.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    
    config = {  
                "bert_model": "monologg/biobert_v1.0_pubmed_pmc", # change model name here
                "freeze_bert": False,
                "maxlen": 25,
                "batch_size": 16,
                "lr": 1e-5,
                "epochs": 2,
             }
    wandb.init(project="BERT-based-Ranking-for-Biomedical-Entity-Normalization", entity="harsh1729", config=config)

    main(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        bert_model="monologg/biobert_v1.0_pubmed_pmc", # change model name here
        freeze_bert=False,
        maxlen=25,
        bs=16,
        device="cuda:0",
        lr=1e-5,
        epochs=5,
        iters_to_accumulate=2,
        )

    wandb.finish() 
