import torchaudio
from models import MultimodalSentimentModel
from pathlib import Path
import cv2
import numpy as np
import subprocess
import torch
import whisper
from transformers import AutoTokenizer

EMOTION_MAP ={0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy',
        4: 'neutral', 5: 'sadness', 6: 'surprise'}
SENTIMENT_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'} 


class VideoProcessor:
    def __init__(self, vid_dir:str) -> None:
        self.vid_dir = vid_dir
        
    def process_video (self, vid_path):
        self.vid_dir = vid_path
        vid_width = 224
        vid_height = 224
        cap = cv2.VideoCapture(self.vid_dir)
        frames = []
        
        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found {self.vid_dir}")
            
            #Reading first video frame for validation
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found {self.vid_dir}")
            
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

class AudioProcessor:
    def __init__(self, vid_path: str, max_length=300) -> None:
        self.vid_path = vid_path
    
    def process_audio(self, vid_path):
        self.vid_dir = vid_path
        audio_path = Path(self.vid_path).with_suffix('.wav')
        
        try:
            subprocess.run([
                'ffmpeg',
                '-i', self.vid_path,
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
            
            # Spectogram stucture: [batch_size, channel, freq, time] -> [batch_size, freq, time] 
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
                
class VideoUtteranceProcessor:
    def __init__(self, vid_path) -> None:
        self.vid_path = vid_path
        self.video_proc = VideoProcessor(vid_path)
        self.audio_proc = AudioProcessor(vid_path)
        
    def extract_segment(self, start_time, end_time, temp_dir="/tmp"):
        Path.mkdir(Path(temp_dir), exist_ok=True)
        segment_path = Path() / temp_dir / f"segment_{start_time}_{end_time}.mp4"
        
        subprocess.run([
            "ffmpeg", "-i", self.vid_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "libx264",
            "-y",
            segment_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not segment_path.exists() or segment_path.stat().st_size == 0:
            raise ValueError(f"Segment extraction failed: {segment_path}")
        
        return segment_path

