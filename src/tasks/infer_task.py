# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from typing import Any, Dict, List, Tuple
import copy
import hydra
import logging
import os
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from PIL import Image
import pandas as pd

import warnings

from submodules.text_recognition_lightning.src import utils
from submodules.text_recognition_lightning.src.datamodules.ndl_components.ndl_dataset import XMLRawAttrWithCli

log = utils.get_pylogger(__name__)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(logging.CRITICAL)

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Lightning couldn't infer the indices fetched for your dataloader.*")


def create_object_dict(cfg: DictConfig) -> dict:
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "trainer": trainer,
    }
    return object_dict


def infer(object_dict: dict, input_data: Dict[str, Any]) -> Dict[str, Any]:
    output_data = copy.deepcopy(input_data)

    cfg = object_dict['cfg']
    model = object_dict['model']
    datamodule = object_dict['datamodule']
    trainer = object_dict['trainer']

    log.info("Starting predict!")
    pil_image = Image.fromarray(input_data['img'])
    pid = os.path.basename(input_data['img_path']).split('_')[0]
    datamodule.set_input_data(pil_image, input_data['xml'], pid)
    preds = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    log.info("Done predict!")
    if len(preds) > 0:
        preds = pd.concat([pd.DataFrame(batch) for batch in preds]).sort_values('file_idx', kind='mergesort')

        xmldata = XMLRawAttrWithCli(output_data, additional_elements=cfg.datamodule.additional_elements)
        xmldata.set_data(output_data['xml'], pid)
        xmldata = iter(xmldata)

        for idx in range(preds.shape[0]):
            line = next(xmldata).attrib
            line['STRING'] = preds["text"].iloc[idx]

    return output_data
