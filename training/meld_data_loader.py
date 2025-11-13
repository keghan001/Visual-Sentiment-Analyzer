import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import cv2
import subprocess
import torchaudio


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
        vid_width = 224
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
            # frames += np.zeros_like(frames[0]) * (30 -len(frames))
            pad_count = 30 - len(frames)
            pad_frames = [np.zeros_like(frames[0]) for _ in range(pad_count)]
            frames.extend(pad_frames)
        else:
            frames = frames[:30]
        
        #Before permute(): [frames, height, width, channels]
        #After permute(): [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
    
    def _extract_audio_features(self, video_dir: Path):
        audio_path = video_dir.with_suffix('.wav')
        
        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_dir,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000', 
                '-ac', '1', 
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
            #loading audio formats with torch
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            #Transforming waveform -> melspectogram
            mel_spectogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )
            
            mel_spec = mel_spectogram(waveform)
            
            #Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
            
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]
            
            return mel_spec
        except subprocess.CalledProcessError as se:
            raise ValueError(f"Audio extraction error: {str(se)}")
        except Exception as e:
            raise ValueError(f"Audio error {str(e)}")
        finally:
            if audio_path.exists():
                audio_path.unlink()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        row = self.data.iloc[index]     # type: ignore
        
        try:
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
            audio_features = self._extract_audio_features(vid_path)
            # print(audio_features)
            
            #Map sentiment and emotion labels
            emotion_label = self.emotion_map.get(row['Emotion'].lower(), 0)
            sentiment_label = self.sentiment_map.get(row['Sentiment'].lower(), 0)
            
            return {
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error processing {vid_path} {str(e)}")
            return None

def collate_fn(batch):
    #Filter out None samples
    batch = list(filter(None, batch))
    return torch.utils.data.default_collate(batch)  

def prepare_dataloaders(train_csv, train_video_dir,
                        dev_csv, dev_video_dir,
                        test_csv, test_video_dir, batch_size=32):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)
    
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)
    
    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn)
    
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader

def main() -> None:
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        Path('../dataset/train/train_sent_emo.csv'), Path('../dataset/train/train_splits'),
        Path('../dataset/dev/dev_sent_emo.csv'), Path('../dataset/dev/dev_splits_complete'),
        Path('../dataset/test/test_sent_emo.csv'), Path('../dataset/test/output_repeated_splits_test'),
    )
    
    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break

if __name__ == "__main__":
    main()