import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from model import SentencePairClassifier
from utils import test_prediction
from data import dataset

def test_bert(
    bert_model,
    path_to_model,
    path_to_output_file,
    df_test,
    bs,
    device,
    maxlen=25,
    dataset_name='dataset name not provided'
):
    print("Reading test data...")
    test_set = dataset(df_test, maxlen=maxlen, bert_model=bert_model)
    test_loader = DataLoader(test_set, batch_size=bs, num_workers=8)

    model = SentencePairClassifier(bert_model=bert_model)

    print()
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)

    print("Predicting on test data...")
    test_prediction(net=model, device=device, dataloader=test_loader, with_labels=True,  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                    result_file=path_to_output_file, dataset_name=dataset_name)
    print()
    print("Predictions are available in : {}".format(path_to_output_file))

    labels_test = df_test['label']  # true labels

    probs_test = pd.read_csv(path_to_output_file).iloc[:,0]  # prediction probabilities
    threshold = 0.5   # you can adjust this threshold for your own dataset
    preds_test=(probs_test>=threshold).astype('uint8') # predicted labels using the above fixed threshold

    accuracy, f1_score_ = accuracy_score(labels_test, preds_test),\
    f1_score(labels_test, preds_test)
    
    print(f"Val Accuracy = {accuracy},\n F1 Score = {f1_score_}")
    return accuracy, f1_score_ 
