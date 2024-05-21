from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset

class AudioSetTrainingSet(Dataset):
    def __init__(self, data_dir_path):
        self.ds = load_dataset(data_dir=data_dir_path)
        self.huggingface_audioset = total
        
    def __len__(self):
        return len(self.huggingface_audioset)

    def __getitem__(self, idx):
        return self.huggingface_audioset[idx]

        