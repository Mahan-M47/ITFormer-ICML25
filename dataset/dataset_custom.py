import torch
from torch.utils.data import Dataset

class CustomTsQaDataset(Dataset):
    def __init__(self, data):
        """
        data: list of dicts
        each dict must contain a 'ts_values' key
        """
        self.data = data

        # convert ts_values to tensors once (not in __getitem__)
        for d in self.data:
            if not torch.is_tensor(d["ts_values"]):
                d["ts_values"] = torch.tensor(d["ts_values"], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
