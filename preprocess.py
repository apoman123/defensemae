from datasets import load_dataset, concatenate_datasets, Audio
import torch
from torchaudio.transforms import MelSpectrogram
torch.set_num_threads(1)
unbalanced = load_dataset("agkphysics/AudioSet", "unbalanced", trust_remote_code=True)['train']
balanced = load_dataset("agkphysics/AudioSet", "balanced", trust_remote_code=True)['train']
total = concatenate_datasets([unbalanced, balanced])
total = total.cast_column("audio", Audio(sampling_rate=16_000))
sample_rate = 16000

win_length = int(sample_rate * 0.025)  # 25ms
hop_length = int(sample_rate * 0.01)  # 10ms
transform = MelSpectrogram(
    sample_rate=sample_rate,
    win_length=win_length,
    hop_length=hop_length,
    n_fft=win_length,
    n_mels=128,
    window_fn=torch.hamming_window
)

# unbalanced = unbalanced.cast_column("audio", Audio(sampling_rate=sample_rate))

def filtering(examples):
    accept = []
    for idx in range(len(examples['audio'])):
        try:
            wav = torch.from_numpy(examples['audio'][idx]['array']).reshape(1, -1).float()
            accept.append(True)
        except:
            print(f'one broken data, video id: {example["video_id"]}')
            accept.append(False)
            
    return accept
        
def pad_and_convert_to_spectrogram(examples):
    spectrograms = []
    for audio_data in examples['audio']:
        wav = torch.from_numpy(audio_data['array']).reshape(1, -1).float()
        # pad small white noise to sample_rate0, which is 10 seconds
        N, L = wav.shape
        if L < sample_rate*10:
            append_len = sample_rate*10 - L 
            wav = torch.cat([wav, torch.randn(1, append_len)*0.001], dim=-1)
        elif L > sample_rate*10:
            wav = wav[:, :sample_rate*10]
            
        spec = transform(wav)
        spectrograms.append(spec)
    examples['spectrogram'] = spectrograms
    return examples
    
# ds = total.map(pad_and_convert_to_spectrogram)
# fisrt filter the broken data out
ds = total.filter(filtering, batched=True, batch_size=1000, num_proc=18)
ds = ds.map(pad_and_convert_to_spectrogram, batched=True, batch_size=1000, num_proc=18)
ds.save_to_disk("/home/apoman123/data/nas07/DataSet/Audio/audioset_full_training_set")
