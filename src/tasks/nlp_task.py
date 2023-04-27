# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from typing import Any, Dict, Tuple
import pathlib

import hydra
from omegaconf import DictConfig
from transformers import Trainer, EvalPrediction
from datasets import Dataset
from evaluate import CombinedEvaluations as CbEval
import torch

from src import utils

log = utils.get_pylogger(__name__)


def hook_argmax(module, inputs, outputs):
    logits = outputs.logits[0] if isinstance(outputs.logits, tuple) else outputs.logits
    outputs.logits = torch.argmax(logits, axis=1)
    return outputs


class CombinedEvaluations(CbEval):
    def __call__(self, p: EvalPrediction):
        return self.compute(predictions=p.predictions, references=p.label_ids)


@utils.task_wrapper
def main_task(cfg: DictConfig) -> Tuple[None, Dict[str, Any]]:
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    dataset: Dataset = hydra.utils.instantiate(cfg.datamodule)

    dataset = dataset.rename_columns({
        cfg.datamodule.text: 'text',
        cfg.target: 'label',
    })

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    model.register_forward_hook(hook_argmax)

    post_model = hydra.utils.instantiate(cfg.callbacks)['post_model']

    log.info(f"Instantiating tokenizer <{cfg.tokenizer._target_}>")
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    dataset = dataset.map(lambda x: tokenizer(x['text']), batched=True)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    compute_metrics = CombinedEvaluations(['accuracy', 'f1', 'precision', 'recall'])
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer,
                                               model=model,
                                               tokenizer=tokenizer,
                                               train_dataset=dataset['train'],
                                               eval_dataset=dataset['val'],
                                               compute_metrics=compute_metrics)

    object_dict = {
        "cfg": cfg,
        "datamodule": dataset,
        "model": model,
        "trainer": trainer,
    }

    if cfg.task_name == 'train':
        trainer.train()
        trainer.save_model()

    if cfg.task_name == 'eval':
        for level in range(1, 5):
            ds = dataset['test'].filter(lambda x: x['LEVEL'] == level)
            if not ds:
                continue

            pred = trainer.predict(ds)
            metrics = pred.metrics
            predictions = pred.predictions

            metrics.update({"test_msec_per_samples": 1000 / metrics['test_samples_per_second']})
            trainer.log_metrics(f"test{level}", metrics)
            trainer.save_metrics(f"test{level}", metrics)
            if post_model.model:
                predictions = post_model(ds.to_pandas(), predictions)
                metrics.update(trainer.compute_metrics.compute(predictions, pred.label_ids))
                trainer.log_metrics(f"test{level}-rf", metrics)
                trainer.save_metrics(f"test{level}-rf", metrics)

            output_prediction_file = pathlib.Path(cfg.paths.output_dir) / f"generated_predictions{level}.txt"
            with open(output_prediction_file, 'w') as f:
                f.write('\n'.join(predictions.astype(str).tolist()))
            output_prediction_file = pathlib.Path(cfg.paths.output_dir) / f"misstaked_predictions{level}.txt"
            with open(output_prediction_file, 'w') as f:
                for preds, texts, labels in zip(predictions, ds['text'], ds['label']):
                    if preds != labels:
                        f.write(f"{preds}\t{labels}\t{texts}\n")
    elif cfg.task_name == 'xml':
        ds = dataset['test']

        pred = trainer.predict(ds)
        metrics = pred.metrics
        predictions = pred.predictions
        if post_model.model:
            predictions = post_model(ds.to_pandas(), predictions)

        from src.datamodules.ndl_components.ndl_dataset import XMLRawAttr
        dataset = iter(XMLRawAttr(XMLRawAttr.find_xml(cfg.datamodule.data_dirs),
                                  pathlib.Path(cfg.paths.output_dir) / 'xml'))
        for idx in range(len(predictions)):
            line = next(dataset).attrib
            if cfg.target in line:
                line[f'{cfg.target}_GT'] = line[cfg.target]
            line[cfg.target] = 'TRUE' if predictions[idx] else 'FALSE'
        for _ in dataset:  # skip to end
            pass

    return metrics, object_dict
