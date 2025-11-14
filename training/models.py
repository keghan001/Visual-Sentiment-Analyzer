from pathlib import Path
import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
from meld_data_loader import MELDDataset

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
        
        self.backbone.fc = nn.Sequential( #type: ignore
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        x = x.transpose(1,2)
        return self.backbone(x)

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


class MultimodalSentimentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(128*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)
        )
        
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3) # Negative, Neutral, Positive
        )
    
    def forward(self, text_input, video_frames, audio_features):
        text_features = self.text_encoder(
            text_input['input_ids'],
            text_input['attention_mask']
        )
        
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)
        
        # Concatenate multimodal features
        combined_features = torch.cat([
            text_features,
            video_features,
            audio_features
        ], dim=1) # [batch_size, 128 * 3]      
        
        # Fusion layer
        fused_feats = self.fusion_layer(combined_features)
        
        emotion_output = self.emotion_classifier(fused_feats)
        sentiment_output = self.sentiment_classifier(fused_feats)
        
        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output
        }
        
class MultimodalTrainer():
    def __init__(self, model, train_loader, val_loader) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        #Log dataset sized
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print("\nDataset sizes: ")
        print(f"Training samples: {train_size:,}")
        print(f"Validation samples: {val_size:,}")
        print(f"Validation samples: {len(train_loader):,}")
        
        #Optimizer initialization
        self.optimizer = torch.optim.Adam([
            {'params': model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': model.emotion_clasifier.parameters(), 'lr': 5e-4},
            {'params': model.sentiment_classifier.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-5)



if __name__ == "__main__":
    dataset = MELDDataset('../dataset/train/train_sent_emo.csv',
                    Path('../dataset/train/train_splits'))
    
    sample = dataset[0]
    
    model = MultimodalSentimentModel()
    model.eval()
    
    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0), #type: ignore
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0) #type: ignore
    }
    
    video_frames = sample['video_frames'].unsqueeze(0)  #type: ignore
    audio_frames = sample['audio_features'].unsqueeze(0) #type: ignore
    
    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_frames)
        
        emotion_preds = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_preds = torch.softmax(outputs['sentiments'], dim=1)[0]
    
    emotion_map = {
        0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy',
        4: 'neutral', 5: 'sadness', 6: 'surprise'
    }
    
    sentiment_map = {
        0: 'negative', 1: 'neutral', 2: 'positive'
    }   
    
    for i, prob in enumerate(emotion_preds):
        print(f"Predicted Emotion: {emotion_map[i]}: {prob:.2f}")
        
    for i, prob in enumerate(sentiment_preds):
        print(f"Predicted Sentiment: {sentiment_map[i]}: {prob:.2f}")