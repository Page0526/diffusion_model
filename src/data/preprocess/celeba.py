import imageio
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import kagglehub

class CelebADataset(Dataset):
    dataset_dir = 'celeba'
    dataset_url = 'https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256'

    def __init__(self, data_dir: str = 'data')-> None:
        super().__init__()
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(self.data_dir) if f.endswith((".jpg", ".png"))]

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.image_files[index])
        img = imageio.v2.imread(img_path)
        
        return img
    
if __name__ == "__main__":

    dataset = CelebADataset("/mnt/banana/student/ptrang/diffusion_model/data/celeba_hq_256")
    print(len(dataset))


    img = dataset[0]
    print(img.shape)

    plt.imshow(img)
    plt.show()
