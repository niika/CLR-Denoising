from pytorch_lightning.loggers import CometLogger
from typing import Any
import pytorch_lightning as pl
from models import CLRAutoencoder
from sidar import SIDAR
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from kornia.losses import ssim_loss
import torchvision
from torchvision import transforms
from pytorch_lightning.tuner import Tuner
from torch import nn 

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

toPIL = transforms.ToPILImage()


class CLRAutoencoderPL(pl.LightningModule):

    def __init__(self, root_dir: str, *args: Any, **kwargs: Any) -> None:
        super().__init__( )

        self.autoencoder = CLRAutoencoder(*args, **kwargs)
        self.batch_size= 64
        data = SIDAR(root_dir, 1)
        n = len(data)

        train, val, test  = (0.8, 0.1, 0.1)
        train = int(train*n)
        val = int(val*n)
        test = n - train - val

        train_data, val_data, test_data = random_split(data, [train,val,test], generator=torch.Generator().manual_seed(45))
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data


        self.loss = nn.L1Loss()
           
    def train_dataloader(self):
        return DataLoader(self.train_data,collate_fn=collate_fn, batch_size=self.batch_size , num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_data,collate_fn=collate_fn, batch_size=self.batch_size , num_workers=16)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,collate_fn=collate_fn, batch_size=self.batch_size , num_workers=16)
        
    def training_step(self, batch, batch_idx):
        """Forward pass 
        
        batch : tuple of tensors of size (B, C, H, W)
        """
        # training_step defined the train loop. It is independent of forward
        x, y = batch

        # Forward pass
        out = self.autoencoder(x)
        loss = self.loss(y, out)

        ssim = ssim_loss(y,out,7)
        #self.logger.experiment.log_metric({'train_loss':loss.item()})

        #seqs = batches2Grids(x)
        #self.logger.experiment.log_image(seqs[0])
        #logs = {'train_loss':loss}
        self.log('train_ssim', ssim)
        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        """ validation step
        
        batch : tuple of tensors of size (B, S, C, H, W)
        """
        # training_step defined the train loop. It is independent of forward
        x, y = batch

        # Forward pass
        out = self.autoencoder(x)
        
        loss = self.loss(y, out)
        ssim = ssim_loss(y,out,7)
        
        self.log('val_ssim', ssim)
        self.log('val_loss', loss)
        
        if batch_idx % 5 == 0:
            grid = torchvision.utils.make_grid(out) 
            try:
                self.logger.experiment.log_image(toPIL(grid), name="reconstructions"+str(batch_idx) , step =self.global_step) 
            except: 
                pass
        
        return {'loss_val': loss, "input": x, "gt":y, "output": out}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=0.0003 )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(self.train_dataloader()), eta_min=0,
                                                           last_epoch=-1)
        #return [optimizer], [lr_scheduler]
        return {
           'optimizer': optimizer,
           'lr_scheduler': lr_scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_loss'
       }



if __name__ == "__main__":

    COMET_ML_PROJECT = "CLR-Restoration"
    experiment_name = "ResNet50-1024"

        # Create Logger
    comet_logger = CometLogger(
    api_key="tMEjeyq5M7v1IPRCvS5fyGyuo",
    workspace="semjon", # Optional
    project_name= COMET_ML_PROJECT, # Optional
    # rest_api_key=os.environ["COMET_REST_KEY"], 
    #save_dir='./hyperparameter',
    experiment_name=experiment_name, # Optional,
    #display_summary_level = 0
)



net = CLRAutoencoderPL(root_dir="../SIDAR",checkpoint="runs/ResNet50-1024/checkpoint_0500.pth.tar")
trainer = pl.Trainer(accelerator="gpu",num_nodes=1,logger=comet_logger, max_epochs=1000, #accumulate_grad_batches=3
                    fast_dev_run=False,
                    )
tuner = Tuner(trainer)
tuner.scale_batch_size(net, mode="binsearch")

trainer.fit(net)
