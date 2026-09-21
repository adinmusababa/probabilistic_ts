import pandas as pd
import torch
from torch.utils.data import Dataset

class EETh1Dataset(Dataset):
    def __init__(
            self,
            csv_path,
            context_length=96,
            prediction_length=24,
            target_column='OT',
            ):

        # inisialisasi variabel self
        self.df = pd.read_csv(csv_path)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_column = target_column

        # variabel numerik kecuali date
        self.features = [
            col for col in self.df.columns if col != "date"
        ]

        self.data = self.df[self.features].values

    # fungsi untuk mendapatkan panjang dataset
    def __len__(self):
        return (
            len(self.data)
            - self.context_length
            - self.prediction_length
            + 1
        )

    # fungsi untuk mendapatkan item dataset berdasarkan indeks
    def __getitem__(self, idx):
        start = idx
        context_end = start + self.context_length
        target_end = context_end + self.prediction_length

        context = self.data[start:context_end]
        target = self.data[context_end:target_end]

        context = torch.tensor(
            context,
            type=torch.float32
        )
        target = torch.tensor(
            target,
            type=torch.float32
        )

        return {
            "context": context,
            "target": target
        }

    