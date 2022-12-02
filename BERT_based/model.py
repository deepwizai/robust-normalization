import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from transformers import AutoModel

class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model="emilyalsentzer/Bio_ClinicalBERT", freeze_bert=True):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)

        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        if bert_model == "emilyalsentzer/Bio_ClinicalBERT": # 108M parameters
            hidden_size = 768
        elif bert_model == "bert-base-cased": # 110M parameters
            hidden_size = 768
        elif bert_model == "dmis-lab/biobert-base-cased-v1.1":
            hidden_size = 768
        elif bert_model == "monologg/biobert_v1.0_pubmed_pmc":
            hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        else:
            for p in self.bert_layer.parameters():
                p.requires_grad = True

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        
    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        cont_reps, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids)['last_hidden_state'],\
        self.bert_layer(input_ids, attn_masks, token_type_ids)['pooler_output']

        # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
        # objective during pre-training.
        x = self.cls_layer(pooler_output)
        x = self.softmax(x)
        return x
