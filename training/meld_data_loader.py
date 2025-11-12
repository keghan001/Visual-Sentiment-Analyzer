import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import cv2


class MELDDataset(Dataset):
    def __init__(self, csv_path: Path, video_dir: Path) -> None:
        self.data = pd.read_csv(csv_path)
        
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        self.emotion_map = {
            'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6
        }
        
        self.sentiment_map = {
            'negative': 0, 'neutral': 1, 'postive': 2
        }
        
    def _load_video_frames (self, video_dir):
        vid_width = 244
        vid_height = 224
        cap = cv2.VideoCapture(video_dir)
        frames = []
        
        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found {video_dir}")
            
            #Reading first video frame for validation
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found {video_dir}")
            
            #Resetting frame index postion to zero 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            #Reading 30 frames from video stream
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                #Resizing, normalizing and appending frames
                frame = cv2.resize(frame, (vid_width, vid_height))
                frame = frame / 255.0
                frames.append(frame)
            
        except Exception as ex:
            raise ValueError(f"Video error {str(ex)}")
        finally:
            cap.release()
            
        if (len(frames) == 0):
            raise ValueError("No frames could be extracted")
        
        #Pad or truncate frames
        if len(frames) < 30:
            frames += np.zeros_like(frames[0]) * (30 -len(frames))
        else:
            frames = frames[:30]
        
        #Before permute(): [frames, height, width, channels]
        #After permute(): [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
        
        vid_path = Path(self.video_dir) / video_filename
        
        if vid_path.exists() == False:
            raise FileNotFoundError(f"Video file {vid_path} not available")
        
        text_inputs = self.tokenizer(row['Utterance'],
                                    padding='max_length',
                                    truncation=True,
                                    max_length=128,
                                    return_tensors='pt')
        
        video_frames = self._load_video_frames(vid_path)
        
        print(video_frames)



if __name__ == "__main__":
    csv_path = Path('../dataset/dev/dev_sent_emo.csv')
    vid_dir = Path('../dataset/dev/dev_splits_complete')
    meld = MELDDataset(csv_path, vid_dir)
    
    print(meld[3])
    