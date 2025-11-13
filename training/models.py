import torch.nn as nn
from transformers import BertModel


class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        #Preventing pretrained model from being trained
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.projection = nn.Linear(768, 128)
        
        def forward(self, input_ids, attention_mask):
            #Extract BERT embeddings
            outputs = self.bert(input_ids= input_ids, attention_mask= attention_mask)
            
            #Use [CLS] token representation
            pooler_out = outputs.pooler_output
            
            return self.projection(pooler_out)