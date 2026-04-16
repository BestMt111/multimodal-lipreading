import logging
import os

import hydra
import torch

from pytorch_lightning import Trainer
from lightning import ModelModule
from datamodule.data_module import DataModule


@hydra.main(version_base="1.3", config_path="configs", config_name="config_subset")
def main(cfg):
    # Set modules and trainer
    if cfg.data.modality in ["audio", "video"]:
        from lightning import ModelModule
    elif cfg.data.modality == "audiovisual":
        from lightning_av import ModelModule
    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    trainer = Trainer(num_nodes=1, gpus=cfg.gpus)
    # Training and testing
    # modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage))
    trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
