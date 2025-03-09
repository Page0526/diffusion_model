import imageio
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from pathlib.Path import Path

class Set14Dataset(Dataset):
    dataset_dir = "set14"
    dataset_url = "https://www.kaggle.com/datasets/msahebi/super-resolution"
    def __init__(self, data_dir:str='data', transform=None):
        super().__init__()
        self.paths = list(Path(data_dir).glob("*/*.png"))

    def load_image(self, index:int)->Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index:int):
        img = self.load_image(index)
        return img
    
if __name__ == '__main__':

    dataset = Set14Dataset("")
    print(len(dataset))

    img = dataset[0]
    print(img.shape)

    plt.imshow(img)
    plt.show()