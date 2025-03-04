import imageio
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class FFHQDataset(Dataset):
    data_dir = 'ffhq'
    data_url = 'https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set'

    def __init__(self, data_dir: str = 'data')-> None:
        super().__init__()
        self.data_dir = os.path.join(data_dir, self.data_dir)
        self.image_files = [f for f in os.listdir(self.data_dir) if f.endswith((".jpg",".png"))]

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.image_files[index])
        img = imageio.v2.imread(img_path)
        
        return img
    
if __name__ == "__main__":

    dataset = FFHQDataset("")
    print(len(dataset))

    img = dataset[0]
    print(img.shape)

    plt.imshow(img)
    plt.show()
