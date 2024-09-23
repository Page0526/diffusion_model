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
                cond: Tensor | None = None):
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: Two tensor of noise
        """
        return self.net(x, cond=cond)

    def model_step(self, batch: Any):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
        """

        batch, cond = batch
        label = cond['label']
        if not isinstance(label, Tensor):
            label = torch.tensor(label, dtype=torch.long)
        # from IPython import embed
        # embed()
        preds, targets = self.forward(batch, cond=label)
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
        
        if batch_idx == -1:
            # generate images
            reals = batch[0]
            fakes = self.net.sample(n_samples=reals.shape[0], device=self.device)

            # transform images and calculate fid
            if preds.shape[1] == 1:
                # from IPython import embed
                # embed()
                # gray to rgb image
                rgb_fakes = torch.cat([fakes, fakes, fakes], dim=1)
                rgb_reals = torch.cat([reals, reals, reals], dim=1)
            else:   
                rgb_fakes = fakes
                rgb_reals = reals
            
            transform_reals = torch.nn.functional.interpolate(rgb_reals,size=(299,299),mode='bilinear')
            transform_fakes = torch.nn.functional.interpolate(rgb_fakes,size=(299,299),mode='bilinear')
            
            '''
            TODO: Need to be normalized to [0,1]
            '''
            normalized_reals = (transform_reals + 1) / 2  # Assuming original images are in range [-1, 1]
            normalized_fakes = (transform_fakes + 1) / 2  # Assuming original images are in range [-1, 1]

            self.fid.update(normalized_fakes,real=False)
            self.fid.update(normalized_reals,real=True)

            # log image on wandb
            reals=make_grid(reals, nrow=8, normalize=True)
            fakes=make_grid(fakes, nrow=8, normalize=True)

            self.logger.log_image(key='val/sample',images=[reals, fakes],caption=['real','fake'])

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
        condition = batch[1]['label'].clone().detach().long()
        fakes = self.net.sample(n_samples=reals.shape[0], cond=condition, device=self.device)

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
        
        
        normalized_reals = (transform_reals + 1) / 2  # Assuming original images are in range [-1, 1]
        normalized_fakes = (transform_fakes + 1) / 2  # Assuming original images are in range [-1, 1]

        self.fid.update(normalized_fakes,real=False)
        self.fid.update(normalized_reals,real=True)

        # log image on wandb
        reals=make_grid(reals, nrow=8, normalize=True)
        fakes=make_grid(fakes, nrow=8, normalize=True)
        # self.logger.experiment.log({
        #     "test/sample": [wandb.Image(reals, caption='reals'), wandb.Image(fakes, caption='fakes')]
        # })
        self.logger.log_image(key='test/sample',images=[reals, fakes],caption=['real','fake']) 

if __name__ == "__main__":
    _ = DiffusionModule(None, None, None, None)