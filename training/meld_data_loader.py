from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer


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
        
        print(text_inputs)



if __name__ == "__main__":
    csv_path = Path('../dataset/dev/dev_sent_emo.csv')
    vid_dir = Path('../dataset/dev/dev_splits_complete')
    meld = MELDDataset(csv_path, vid_dir)
    
    print(meld[1])
    