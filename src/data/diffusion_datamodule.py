import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Any, Dict, Optional, Tuple
from src.data.datasets.mri import MRIDataset
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision.transforms import transforms
from src.data.datasets.__init__ import init_dataset
import h5py as h5

class DiffusionDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_dir: Tuple[str, str, str] = None,
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 4,
        dataset_name: str = 'mri',
        pin_memory: bool = False,
        image_size: int = None,
    ) -> None:
        
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        pass


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train or self.data_val or self.data_test:
            return
            
        # Initialize datasets
        if self.hparams.train_val_test_dir:
            # Load separate train/val/test datasets
            train_dir, val_dir, test_dir = self.hparams.train_val_test_dir
            self.data_train = init_dataset(self.hparams.dataset_name, data_dir=self.hparams.data_dir, train_val_test_dir=train_dir)
            self.data_val   = init_dataset(self.hparams.dataset_name, data_dir=self.hparams.data_dir, train_val_test_dir=val_dir)
            self.data_test  = init_dataset(self.hparams.dataset_name, data_dir=self.hparams.data_dir, train_val_test_dir=test_dir)

        else:
            if self.hparams.dataset_name == 'mri':
                self.data_train = MRIDataset(mode='train', dataset_path=self.hparams.data_dir)
                self.data_test = MRIDataset(mode='tst', dataset_path=self.hparams.data_dir)
            else:
                # Load full dataset and random split
                full_dataset = init_dataset(self.hparams.dataset_name, data_dir=self.hparams.data_dir)

                # Optional: limit size for quick testing
                total_len = sum(self.hparams.train_val_test_split)
                if 1 < total_len < len(full_dataset):
                    full_dataset = Subset(full_dataset, range(total_len))

                self.data_train, self.data_val, self.data_test = random_split(
                    full_dataset,
                    lengths=self.hparams.train_val_test_split,
                    generator=torch.Generator().manual_seed(42)
                )


    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.data_val is None:
            return []  # ⚡ If no val set, Lightning will skip validation
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    data_dir = '/mnt/banana/student/ptrang/diffusion_model/data/mri/dataset.hdf5'
    batch_size = 1
    train_val_test_split = [0.8, 0, 0.2]
    dataset_name = "mri"

    dm = DiffusionDataModule(data_dir=data_dir, 
                             batch_size=batch_size, 
                             train_val_test_split=train_val_test_split, 
                             dataset_name=dataset_name)
    
    dm.prepare_data()
    dm.setup(stage="fit")

    # Get DataLoaders
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()


    # Fetch one batch
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    from IPython import embed
    embed()

    # Assertions to check things are not broken
    assert len(train_batch) > 0, "Train batch is empty!"
    assert len(test_batch) > 0, "Test batch is empty!"
    print("✅ DiffusionDataModule test passed for MRIDataset!")



