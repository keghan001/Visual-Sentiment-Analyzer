import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models

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
        
class VideoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)
        
        #Preventing pretrained model from being trained
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_fts = self.backbone.fc.in_features
        
        self.head = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        x = x.transpose(1,2)
        features = self.backbone(x)
        return self.head(features)
    

class AudioEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
            
        )
        
        for param in self.conv_layers.parameters():
            param.requires_grad = False
        
        self.proj_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        x = x.squeeze(1)
        
        features = self.conv_layers(x)
        # Features output: [batch_size, 128, 1]
        
        return self.proj_layers(features.squeeze(-1))