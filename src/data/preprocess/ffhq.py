import imageio
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class FFHQDataset(Dataset):
    data_dir = 'ffhq'
    data_url = 'https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set'

    def __init__(self, data_dir: str = 'data', transform=None)-> None:
        super().__init__()
        self.paths = list(Path(data_dir).glob("*/*.png"))
        self.transform = transform

    def prepare_data(self) -> None:
        pass

    def load_image(self, index: int)-> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img = self.load_image(index)
        return img
    
if __name__ == "__main__":

    dataset = FFHQDataset("")
    print(len(dataset))

    img = dataset[0]
    print(img.shape)

    plt.imshow(img)
    plt.show()
