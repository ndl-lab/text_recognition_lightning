# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from typing import Any, Dict
import copy
import pathlib
import hydra
from omegaconf import DictConfig
from transformers import Trainer
import warnings

from submodules.text_recognition_lightning.src import utils
from submodules.text_recognition_lightning.src.datamodules.xml_datamodule_without_image import from_tree
from submodules.text_recognition_lightning.src.datamodules.ndl_components.ndl_dataset import XMLRawAttrWithCli
from submodules.text_recognition_lightning.src.tasks.nlp_task import hook_argmax

log = utils.get_pylogger(__name__)

warnings.filterwarnings("ignore", ".*is ill-defined and being set to 0.0*")


def create_object_dict(cfg: DictConfig, ckpt_path_title, ckpt_path_author) -> dict:
    log.info(f"Instantiating model <{cfg.model._target_}>")
    cfg['ckpt_path'] = ckpt_path_title
    model_title = hydra.utils.instantiate(cfg.model)
    model_title.register_forward_hook(hook_argmax)
    cfg['ckpt_path'] = ckpt_path_author
    model_author = hydra.utils.instantiate(cfg.model)
    model_author.register_forward_hook(hook_argmax)

    log.info(f"Instantiating tokenizer <{cfg.tokenizer._target_}>")
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    trainer_title: Trainer = hydra.utils.instantiate(cfg.trainer,
                                                     model=model_title,
                                                     tokenizer=tokenizer)
    trainer_author: Trainer = hydra.utils.instantiate(cfg.trainer,
                                                      model=model_author,
                                                      tokenizer=tokenizer)

    object_dict = {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "model_title": model_title,
        "trainer_title": trainer_title,
        "model_author": model_author,
        "trainer_author": trainer_author,
    }

    return object_dict


def infer(object_dict: dict, input_data: Dict[str, Any]) -> Dict[str, Any]:
    output_data = copy.deepcopy(input_data)

    cfg = object_dict['cfg']
    tokenizer = object_dict['tokenizer']
    trainer_title = object_dict['trainer_title']
    trainer_author = object_dict['trainer_author']

    if cfg.task_name != 'xml':
        log.error(f"<<{cfg.task}>> is unknown task")
        return output_data

    dataset = from_tree(input_data['xml'], cfg['datamodule']['text'])
    if dataset['test'].num_rows == 0:
        return output_data
    dataset = dataset.rename_columns({
        cfg.datamodule.text: 'text',
    })
    ds = dataset['test']
    ds = ds.map(lambda x: tokenizer(x['text']), batched=True)
    pid = pathlib.Path(input_data['img_path']).name.split('_')[0]

    predictions_title = trainer_title.predict(ds).predictions
    predictions_author = trainer_author.predict(ds).predictions

    xmldata = XMLRawAttrWithCli('')
    xmldata.set_data(output_data['xml'], pid)
    xmldata = iter(xmldata)

    for idx in range(len(predictions_title)):
        line = next(xmldata).attrib
        line['TITLE'] = 'TRUE' if predictions_title[idx] else 'FALSE'
        line['AUTHOR'] = 'TRUE' if predictions_author[idx] else 'FALSE'

    return output_data
