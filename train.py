import os
import hydra
import logging

from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from avg_ckpts import ensemble
from scripts.change_state_dict import read_pretrained_model
from datamodule.data_module import DataModule

@hydra.main(version_base="1.3", config_path="configs", config_name="config_subset")
def main(cfg):
    seed_everything(42, workers=True)
    # cfg.gpus = torch.cuda.device_count()
    # cfg.gpus =
    logger_path = os.path.join(cfg.exp_dir, cfg.exp_name)
    logger = TensorBoardLogger(save_dir=logger_path, name=cfg.log_name)
    checkpoint = ModelCheckpoint(
        monitor="monitoring_step",
        mode="max",
        dirpath=os.path.join(cfg.exp_dir, cfg.exp_name) if cfg.exp_dir else None,
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    # Set modules and trainer
    if cfg.data.modality in ["audio", "video"]:
        from lightning import ModelModule
    elif cfg.data.modality == "audiovisual":
        from lightning_av import ModelModule

    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    trainer = Trainer(
        **cfg.trainer,
        # val_check_interval=1,  # 0.5 表示每个 epoch 进行一半次数的验证
        # logger=WandbLogger(name=cfg.exp_name, project="auto_avsr"),
        callbacks=callbacks,
        strategy=DDPPlugin(find_unused_parameters=False),
        logger=logger
    )

    trainer.fit(model=modelmodule, datamodule=datamodule)
    ensemble(cfg)

    # testing
    # test_trainer = Trainer(
    #     num_nodes=1,
    #     gpus=cfg.trainer.gpus
    # )
    # ckpt_path = os.path.join(cfg.exp_dir, cfg.exp_name, 'last.ckpt')
    # read_pretrained_model(
    #     cfg,
    #     modelmodule.model,
    #     mode='test',
    #     path=ckpt_path
    # )
    # test_trainer.test(model=modelmodule, datamodule=datamodule)

if __name__ == "__main__":
    main()
