from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import MNIST
import numpy as np
import os.path as osp
import torch

class MnistDataset(Dataset):

    dataset_dir = 'mnist'
    
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    def __init__(self, data_dir: str='data') -> None:
        super().__init__()
        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.prepare_data()
        
    def prepare_data(self) -> None:
        trainset = MNIST(root=self.dataset_dir, # where to download data to?
                        train=True, # get training data
                        download=True, # download data if it doesn't exist on disk
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),  # Resize images if necessary
                            transforms.ToTensor(),  # Convert NumPy arrays to PyTorch tensors
                            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
                        ]), # images come as PIL format, we want to turn into Torch tensors
                        target_transform=None # you can transform labels as well
        )

        testset = MNIST(root=self.dataset_dir, # where to download data to?
                        train=False, # get training data
                        download=True, # download data if it doesn't exist on disk
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),  # Resize images if necessary
                            transforms.ToTensor(),  # Convert NumPy arrays to PyTorch tensors
                            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
                        ]), # images come as PIL format, we want to turn into Torch tensors
                        target_transform=None # you can transform labels as well
        )
        self.dataset = ConcatDataset(datasets=[trainset, testset])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        label_tensor = torch.tensor(label, dtype=torch.long)
        return np.array(image), {'label': label_tensor}

if __name__ == "__main__":
    dataset = MnistDataset(data_dir='data')
    image, cond = dataset[0]
    label = cond['label']
    label_type = type(label)
    # to check if label is already a tensor
    label_shape = label.shape if isinstance(label, torch.Tensor) else 'Not a Tensor'
    print(label_shape)
    