from typing import Any, Dict, Tuple
from torch import Tensor
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchvision.utils import make_grid
from torchmetrics.image import FrechetInceptionDistance
from src.models.diffusion_module import DiffusionModule
from src.models.diffusion.net.condition_diffusion_model import ConditionDiffusionModel

class ConditionDiffusionModule(DiffusionModule):

    def __init__(
        self,
        net: ConditionDiffusionModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None: 
        
        super().__init__(net, optimizer, scheduler, compile)

    def forward(self,
                x: Tensor,
                label: Tensor | None = None):
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: Two tensor of noise
        """
        return self.net(x, label=label)

    def model_step(self, batch: Any):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
        """

        batch, cond = batch
        label = cond['label']
        # print(f"label:{type(label)}")
        if not isinstance(label, Tensor):
            label = torch.tensor(label, dtype=torch.long)
        preds, targets = self.forward(batch, label=label)
        loss = self.criterion(preds, targets)
        return loss, preds, targets
    
    def validation_step(self, batch: Tuple[Tensor, Tensor],
                        batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)

        self.log("val/loss",
                 self.val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        
        # generate images
        reals = batch[0]
        condition = batch[1].get('condition')
        fakes = self.net.sample(n_samples=reals.shape[0], labels=condition, device=self.device)

        # transform images and calculate fid
        if preds.shape[1] == 1:
            # gray to rgb image
            rgb_fakes = torch.cat([fakes, fakes, fakes], dim=1)
            rgb_reals = torch.cat([reals, reals, reals], dim=1)
        else:   
            rgb_fakes = fakes
            rgb_reals = reals
            
        transform_reals = torch.nn.functional.interpolate(rgb_reals,size=(299,299),mode='bilinear')
        transform_fakes = torch.nn.functional.interpolate(rgb_fakes,size=(299,299),mode='bilinear')
        
        if batch_idx%4 == 0:
            self.fid.update(transform_fakes,real=False)
            self.fid.update(transform_reals,real=True)

        # log image on wandb
        reals=make_grid(reals, nrow=8, normalize=True)
        fakes=make_grid(fakes, nrow=8, normalize=True)
        # self.logger.experiment.log({
        #     "test/sample": [wandb.Image(reals, caption='reals'), wandb.Image(fakes, caption='fakes')]
        # })
        self.logger.log_image(key='val/sample',images=[reals, fakes],caption=['real','fake'])

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.log("val/fid",self.fid.compute(), prog_bar=False)
        self.fid.reset()
        pass

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss",
                 self.test_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        # generate images
        reals = batch[0]
        condition = batch[1].get('condition')
        fakes = self.net.sample(n_samples=reals.shape[0], labels=condition, device=self.device)

        # transform images and calculate fid
        if preds.shape[1] == 1:
            # gray to rgb image
            rgb_fakes = torch.cat([fakes, fakes, fakes], dim=1)
            rgb_reals = torch.cat([reals, reals, reals], dim=1)
        else:
            rgb_fakes = fakes
            rgb_reals = reals
            
        transform_reals = torch.nn.functional.interpolate(rgb_reals,size=(299,299),mode='bilinear')
        transform_fakes = torch.nn.functional.interpolate(rgb_fakes,size=(299,299),mode='bilinear')
        
        if batch_idx%4 == 0:
            self.fid.update(transform_fakes,real=False)
            self.fid.update(transform_reals,real=True)

        # log image on wandb
        reals=make_grid(reals, nrow=8, normalize=True)
        fakes=make_grid(fakes, nrow=8, normalize=True)
        # self.logger.experiment.log({
        #     "test/sample": [wandb.Image(reals, caption='reals'), wandb.Image(fakes, caption='fakes')]
        # })
        self.logger.log_image(key='test/sample',images=[reals, fakes],caption=['real','fake'])

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log("test/fid",self.fid.compute(), prog_bar=False)
        self.fid.reset()
        pass

if __name__ == "__main__":
    _ = DiffusionModule(None, None, None, None)