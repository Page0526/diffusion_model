from typing import Any, Dict, Tuple
from torch import Tensor
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchvision.utils import make_grid
from torchmetrics.image import FrechetInceptionDistance, StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchvision import transforms
from src.models.diffusion.net.diffusion_model import DiffusionModel
from src.models.modl.modl import MoDLModel

class MoDLModule(LightningModule):

    def __init__(
        self,
        net: MoDLModel,
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

        self.val_psnr = PeakSignalNoiseRatio()
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.test_psnr = PeakSignalNoiseRatio()
        self.test_ssim = StructuralSimilarityIndexMeasure()


    def forward(self,
                x: Tensor, csm: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return self.net(x, csm, mask) # reconstruction
    
    def model_step(
            self, batch: Tuple[Tensor,
                               Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        
        x, y, csm, mask = batch
        rec = self.forward(x, csm, mask)
        loss = self.criterion(rec, y)
        return loss, rec

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int) -> Tensor:
        _, org, _, _ = batch
        loss, rec = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)

        psnr_value = self.val_psnr(org, rec)
        ssim_value = self.val_ssim(org, rec)

        reals=make_grid(org, nrow=8, normalize=True)
        fakes=make_grid(rec, nrow=8, normalize=True)
        self.logger.log_image(key='train/sample',images=[reals, fakes],caption=['real','fake'])

        self.log("train/loss",
                self.train_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True)
        self.log("train/psnr", 
                psnr_value, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True)
        self.log("train/ssim", 
                ssim_value, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True)
        
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[Tensor, Tensor],
                        batch_idx: int) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        pass
        

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        _, org, _, _ = batch
        loss, rec = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        
        psnr_value = self.test_psnr(rec, org)
        ssim_value = self.test_ssim(rec, org)

        reals=make_grid(org, nrow=8, normalize=True)
        fakes=make_grid(rec, nrow=8, normalize=True)
        self.logger.log_image(key='test/sample',images=[reals, fakes],caption=['real','fake'])

        self.log("test/loss",
                 self.test_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log("test/psnr", 
                 psnr_value, 
                 on_step=False,
                 on_epoch=True, 
                 prog_bar=True)
        self.log("test/ssim", 
                 ssim_value, 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MoDLModule(None, None, None, None)
