# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from typing import Any, Dict, Tuple
import pathlib

import hydra
from omegaconf import DictConfig
from datasets import Dataset
from evaluate import CombinedEvaluations

from src import utils

log = utils.get_pylogger(__name__)


class Evaluations:
    from transformers.trainer_pt_utils import log_metrics, metrics_format

    def is_world_process_zero(self):
        return True

    def __init__(self, num_samples, evaluation_modules):
        import time
        self.num_samples = num_samples
        self.ce = CombinedEvaluations(evaluation_modules)
        self.start_time = time.time()

    def compute(self, predictions=None, references=None, **kwargs):
        from transformers.trainer_utils import speed_metrics
        metrics = self.ce.compute(predictions, references, **kwargs)
        metrics.update(
            speed_metrics(
                'test',
                self.start_time,
                num_samples=self.num_samples,
            )
        )
        metrics.update({"test_msec_per_samples": 1000 / metrics['test_samples_per_second']})
        return metrics


@utils.task_wrapper
def main_task(cfg: DictConfig) -> Tuple[None, Dict[str, Any]]:
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    dataset: Dataset = hydra.utils.instantiate(cfg.datamodule)

    dataset = dataset.rename_columns({
        cfg.datamodule.text: 'text',
        cfg.target: 'label',
    })

    log.info(f"Instantiating model <{cfg.model._target_}>")
    trainer = hydra.utils.instantiate(cfg.model)

    object_dict = {
        "cfg": cfg,
        "datamodule": dataset,
        "model": trainer,
    }

    if cfg.task_name == 'train':
        import joblib
        y_train = dataset['train']['label']
        x_train = dataset['train'].remove_columns(['text', 'label']).to_pandas()
        trainer.fit(x_train, y_train)
        joblib.dump(trainer, pathlib.Path(cfg.paths.output_dir) / "model.pkl")

    for level in range(1, 5):
        ds = dataset['test'].filter(lambda x: x['LEVEL'] == level)
        if len(ds) == 0:
            continue
        y_test = ds['label']
        x_test = ds.remove_columns(['text', 'label', 'LEVEL']).to_pandas()
        pred = trainer.predict(x_test)

        evaluations = Evaluations(
            num_samples=len(y_test),
            evaluation_modules=['accuracy', 'f1', 'precision', 'recall'])

        metrics = evaluations.compute(pred, y_test)
        evaluations.log_metrics(f'eval{level}', metrics)

        output_prediction_file = pathlib.Path(cfg.paths.output_dir) / f"generated_predictions{level}.txt"
        with open(output_prediction_file, 'w') as f:
            f.write('\n'.join(pred.astype(str).tolist()))
        output_prediction_file = pathlib.Path(cfg.paths.output_dir) / f"misstaked_predictions{level}.txt"
        with open(output_prediction_file, 'w') as f:
            for preds, texts, labels in zip(pred, ds['text'], ds['label']):
                if preds != labels:
                    f.write(f"{preds}\t{labels}\t{texts}\n")

    return metrics, object_dict
