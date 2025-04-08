from typing import Any, Dict, Tuple
from torch import Tensor
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchvision.utils import make_grid
from torchmetrics.image import FrechetInceptionDistance, StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchvision import transforms
from src.models.diffusion.net.diffusion_model import DiffusionModel

class DiffusionModule(LightningModule):

    def __init__(
        self,
        net: DiffusionModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        
        super().__init__()

        
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

    
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        
        self.fid = FrechetInceptionDistance(normalize=True)
        self.val_psnr = PeakSignalNoiseRatio()
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.test_psnr = PeakSignalNoiseRatio()
        self.test_ssim = StructuralSimilarityIndexMeasure()


    def forward(self,
                x: Tensor) -> Tuple[Tensor, Tensor]:
        
        preds, targets = self.net(x)
        return preds, targets
    
    def model_step(
            self, batch: Tuple[Tensor,
                               Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        
        batch, _ = batch
        preds, targets = self.forward(batch)
        loss = self.criterion(preds, targets)
        return loss, preds, targets

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.fid.reset()

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int) -> Tensor:
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)

        self.log("train/loss",
                 self.train_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[Tensor, Tensor],
                        batch_idx: int) -> None:
        
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)

        self.log("val/loss",
                 self.val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        
        # generate images
        orig = batch[0]
        reconstr = self.net.sample(n_samples=reals.shape[0], device=self.device)

        psnr_value = self.val_psnr(orig, reconstr)
        ssim_value = self.val_ssim(orig, reconstr)

        if batch_idx == 0:
            # transform images and calculate fid
            if preds.shape[1] == 1:
                # gray to rgb image
                rgb_fakes = torch.cat([reconstr, reconstr, reconstr], dim=1)
                rgb_reals = torch.cat([orig, orig, orig], dim=1)
            else:   
                rgb_fakes = reconstr
                rgb_reals = orig
            
            transform_reals = torch.nn.functional.interpolate(rgb_reals,size=(299,299),mode='bilinear')
            transform_fakes = torch.nn.functional.interpolate(rgb_fakes,size=(299,299),mode='bilinear')
            
            normalized_reals = (transform_reals + 1) / 2  # Assuming original images are in range [-1, 1]
            normalized_fakes = (transform_fakes + 1) / 2  # Assuming original images are in range [-1, 1]

            self.fid.update(normalized_fakes,real=False)
            self.fid.update(normalized_reals,real=True)

            # log image on wandb
            reals=make_grid(reals, nrow=8, normalize=True)
            fakes=make_grid(fakes, nrow=8, normalize=True)
            
            self.logger.log_image(key='val/sample',images=[reals, fakes],caption=['real','fake'])

        
        self.log("val/psnr", psnr_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        if len(self.fid.real_features_sum) > 0 and len(self.fid.fake_features_sum) > 0:
            self.log("val/fid",self.fid.compute())
            self.fid.reset()
        

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        
        # generate images
        orig = batch[0]
        reconstr = self.net.sample(n_samples=reals.shape[0], device=self.device)
        psnr_value = self.test_psnr(reconstr, orig)
        ssim_value = self.test_ssim(reconstr, orig)

        # transform images and calculate fid
        if preds.shape[1] == 1:
            # gray to rgb image
            rgb_fakes = torch.cat([reconstr, reconstr, reconstr], dim=1)
            rgb_reals = torch.cat([orig, orig, orig], dim=1)
        else:
            rgb_fakes = reconstr
            rgb_reals = orig
            
        transform_reals = torch.nn.functional.interpolate(rgb_reals,size=(299,299),mode='bilinear')
        transform_fakes = torch.nn.functional.interpolate(rgb_fakes,size=(299,299),mode='bilinear')
        
        normalized_reals = (transform_reals + 1) / 2  # Assuming original images are in range [-1, 1]
        normalized_fakes = (transform_fakes + 1) / 2  # Assuming original images are in range [-1, 1]

        self.fid.update(normalized_fakes,real=False)
        self.fid.update(normalized_reals,real=True)

        # log image on wandb
        reals=make_grid(reals, nrow=8, normalize=True)
        fakes=make_grid(fakes, nrow=8, normalize=True)

        self.log("test/loss",
                 self.test_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log("test/psnr", psnr_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=True)
        self.logger.log_image(key='test/sample',images=[reals, fakes],caption=['real','fake'])

    def on_test_epoch_end(self) -> None:
        if len(self.fid.real_features_sum) > 0 and len(self.fid.fake_features_sum) > 0:
            self.log("test/fid",self.fid.compute(), prog_bar=False)
            self.fid.reset()

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
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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


if __name__ == "__main__":
    _ = DiffusionModule(None, None, None, None)
