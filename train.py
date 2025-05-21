import subprocess
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset_utils import CAFNetTrainDataset
from net.model import CATNet
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class CATNetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = CATNet(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        ED_result0,ED_result1,ED_result2,restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)+0.5*self.loss_fn(ED_result0, clean_patch)+0.4*self.loss_fn(ED_result1, clean_patch)+0.3*self.loss_fn(ED_result2, clean_patch)

        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150,eta_min = 1e-6)

        return [optimizer],[scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = CAFNetTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    model = CATNetModel()
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()



