import torch
import torch.utils
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self):
        self.x = [[73, 80],
                  [93, 88],
                  [89, 91],
                  [96, 98],
                  [73, 66]]
        self.y = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        y = torch.FloatTensor(self.y[idx])
        return x, y
