import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pathlib.Path as Path
from PIL import Image

class CelebADataset(Dataset):
    dataset_dir = 'celeba'
    dataset_url = 'https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256'

    def __init__(self, data_dir: str = 'data', transform=None)-> None:
        super().__init__()
        self.paths = list(Path(data_dir).glob("*/*.jpg"))

    def prepare_data(self) -> None:
        pass

    def load_image(self, index:int)->Image.image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img = self.load_image(index)
        return img
    
if __name__ == "__main__":

    dataset = CelebADataset("/mnt/banana/student/ptrang/diffusion_model/data/celeba_hq_256")
    print(len(dataset))


    img = dataset[0]
    print(img.shape)

    plt.imshow(img)
    plt.show()
