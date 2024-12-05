from typing import Any, Tuple, Dict

import torch
from torch import Tensor
import pyrootutils
import torch.nn as nn
import lightning as L

from torchmetrics import MeanMetric
from torch.optim import Optimizer, lr_scheduler
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.nn import MSELoss
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss, DiceLoss

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vae.net import BaseVAE


class VAEModule(L.LightningModule):

    def __init__(
        self,
        net: BaseVAE,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
        criterion: nn.Module,
        compile: bool = False,
    ) -> None:
        """_summary_

        Args:
            net (BaseVAE): _description_
            optimizer (Optimizer): _description_
            scheduler (lr_scheduler): _description_
            criterion (nn.Module): _description_
            use_ema (bool, optional): _description_. Defaults to False.
            compile (bool, optional): _description_. Defaults to False.
        """
        
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # VAE
        self.net = net


        assert isinstance(criterion, (MSELoss, SoftBCEWithLogitsLoss)), \
            NotImplementedError(f"only implemented for [MSELoss, SoftBCEWithLogitsLoss]")
        
        # loss function
        self.criterion = criterion

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_psnr = PeakSignalNoiseRatio()
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.test_psnr = PeakSignalNoiseRatio()
        self.test_ssim = StructuralSimilarityIndexMeasure()

    def on_train_batch_end(self, *args, **kwargs):
        pass

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of images
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_psnr.reset()
        self.val_ssim.reset()

    def rescale(self, image):
        # convert range of image from [-1, 1] to [0, 1]
        return image * 0.5 + 0.5

    def model_step(
        self, 
        batch: Tuple[Tensor, Tensor],
    ) -> Dict[str, Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
        """
        targets, _ = batch # img pixel & label
        preds, losses = self.forward(targets)

        if isinstance(self.criterion, (SoftBCEWithLogitsLoss, DiceLoss)):
            targets = self.rescale(targets)

        if losses is None:
            return {"recons_loss": self.criterion(preds, targets)}

        
        losses["recons_loss"] = self.criterion(preds, targets)
        return preds, losses

    def training_step(self, batch: Tuple[Tensor, Tensor],
                    batch_idx: int) -> Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        preds, losses = self.model_step(batch)
    
        # update and log metrics
        loss = sum(losses.values())
        self.train_loss(loss)


        self.log("train/loss",
                self.train_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True)

        for key in losses.keys():
            self.log(f"train/{key}_loss",
                    losses[key].detach(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True)
        
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[Tensor, Tensor],
                        batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        reconstr, losses = self.model_step(batch)

        # update and log metrics
        loss = sum(losses.values())
        self.val_loss(loss)

        self.log("val/loss",
                self.val_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True)

        for key in losses.keys():
            self.log(f"val/{key}",
                    losses[key].detach(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True)
            
        x, y = batch
        psnr_value = self.val_psnr(reconstr, x)
        ssim_value = self.val_ssim(reconstr, x)
        if batch_idx %10 == 0:
            self.logger.log_image(key='val/image', images=[reconstr, x], caption=['reconstruction', 'real'])
        self.log("val/psnr", psnr_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        reconstr, losses = self.model_step(batch)

        # update and log metrics
        loss = sum(losses.values())
        self.test_loss(loss)

        self.log("test/loss",
                self.test_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True)

        for key in losses.keys():
            self.log(f"test/{key}_loss",
                    losses[key].detach(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True)

        x, y = batch
        psnr_value = self.test_psnr(reconstr, x)
        ssim_value = self.test_ssim(reconstr, x)
        if batch_idx %10 == 0:
            self.logger.log_image(key='test/image', images=[reconstr, x], caption=['reconstruction', 'real'])
        self.log("test/psnr", psnr_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
