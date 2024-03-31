import os
import torch

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        self.data_path = data_path
        self.train = train
        self.transform = transform

        self.data = torch.load(os.path.join(self.data_path, "train.pt" if train else "test.pt"), map_location=torch.device("cpu"))
        self.targets = self.data["targets"]
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        image = self.data["images"][idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, target

if __name__ == "__main__":
    for dataset_name in ["AMD", "PALM", "UWF"]:
        print(f"Testing {dataset_name} dataset...")
        dataset = CustomDataset(dataset_name)
        print(len(dataset))
        print(dataset[0])
        print(dataset.targets)

        dataset = CustomDataset(dataset_name, train=False)
        print(len(dataset))
        print(dataset[0])
        print(dataset.targets)
