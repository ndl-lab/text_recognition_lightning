# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from typing import Any, Dict, Tuple
import copy
import pathlib
from omegaconf import DictConfig
import warnings
import joblib

from src import utils
from ..datamodules.ndl_components.ndl_dataset import XMLRawAttrWithCli
from ..datamodules.xml_datamodule_without_image import from_tree


log = utils.get_pylogger(__name__)

warnings.filterwarnings("ignore", ".*Trying to unpickle estimator.*")


def create_object_dict(cfg: DictConfig, pkl_path_title, pkl_path_author):
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    trainer_title = joblib.load(pkl_path_title)
    trainer_author = joblib.load(pkl_path_author)

    object_dict = {
        "cfg": cfg,
        "trainer_title": trainer_title,
        "trainer_author": trainer_author,
    }

    return object_dict


def infer(object_dict: dict, input_data) -> Tuple[None, Dict[str, Any]]:
    output_data = copy.deepcopy(input_data)

    cfg = object_dict['cfg']
    trainer_title = object_dict['trainer_title']
    trainer_author = object_dict['trainer_author']

    if cfg.task_name != 'xml':
        log.error(f"<<{cfg.task}>> is unknown task")
        return output_data

    dataset = from_tree(input_data['xml'], cfg['datamodule']['text'])
    if dataset['test'].num_rows == 0:
        return output_data

    ds = dataset['test']
    removed_columns_list = list(set(ds.column_names) - set(trainer_title.feature_names_in_))
    x_test = ds.remove_columns(removed_columns_list).to_pandas()
    x_test = x_test[trainer_title.feature_names_in_]

    pid = pathlib.Path(input_data['img_path']).name.split('_')[0]

    pred_title = trainer_title.predict(x_test)
    pred_author = trainer_author.predict(x_test)

    xmldata = XMLRawAttrWithCli('')
    xmldata.set_data(output_data['xml'], pid)
    xmldata = iter(xmldata)

    for idx in range(len(pred_title)):
        line = next(xmldata).attrib
        line['TITLE'] = 'TRUE' if pred_title[idx] else 'FALSE'
        line['AUTHOR'] = 'TRUE' if pred_author[idx] else 'FALSE'

    return output_data
