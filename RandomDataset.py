import torch
from torch.utils.data import Dataset

class RandomData(Dataset):
    def __init__(self,input_size,samples) -> None:
        self.samples = samples
        self.input_size = input_size

        self.X = torch.randn((samples,input_size))
        self.Y = torch.randn((samples,1))

    def __len__(self):
        return self.samples

    def __getitem__(self, index) :
        return (self.X[index],self.Y[index])