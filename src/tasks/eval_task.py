# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from typing import Any, Dict, List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[None, Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

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
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.task == "eval":
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    elif cfg.task in ["xml", "render"]:
        render = hydra.utils.instantiate(cfg.render)

        log.info("Starting predict!")
        preds = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        log.info("Done predict!")
        import pandas as pd
        preds = pd.concat([pd.DataFrame(batch) for batch in preds]).sort_values('file_idx', kind='mergesort')

        if cfg.task == "render":
            from ..datamodules.ndl_components.ndl_dataset import XMLRawDataset
            import itertools
            dataset = itertools.chain(*[
                iter(XMLRawDataset(xml_files=dataset.xml_files, additional_elements=cfg.datamodule.additional_elements))
                for dataset in datamodule.data_pred
            ])

            for idx in range(preds.shape[0]):
                image, text, data = next(dataset)
                render(image, text, preds["text"].iloc[idx], preds["pid"].iloc[idx])
        elif cfg.task == "xml":
            from ..datamodules.ndl_components.ndl_dataset import XMLRawAttr
            import itertools
            dataset = itertools.chain(*[
                iter(XMLRawAttr(dataset.xml_files, cfg.render.output_dir,
                                additional_elements=cfg.datamodule.additional_elements))
                for dataset in datamodule.data_pred
            ])
            for idx in range(preds.shape[0]):
                line = next(dataset).attrib
                if 'STRING' in line:
                    line['STRING_GT'] = line['STRING']
                    a_str = line.get('STRING', '')
                    a, b, _ = render(a_str, preds["text"].iloc[idx])
                    print('--', preds["text"].iloc[idx] == a_str)
                    print(a)
                    print(b)
                line['STRING'] = preds["text"].iloc[idx]
            for _ in dataset:  # skip to end
                pass
    else:
        log.error(f"<<{cfg.task}>> is unknown task")

    metric_dict = trainer.callback_metrics
    return metric_dict, object_dict
